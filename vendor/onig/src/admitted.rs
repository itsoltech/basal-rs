//! basal-rs addition: a [`Regex`] whose retained compiled backing is charged
//! to a caller account until the regex is dropped.
//!
//! The admitted constructor compiles exactly like [`Regex::new`]: the same
//! `Syntax::default()` read, options, UTF-8 encoding, compile lock and native
//! compiler. Only the regex-owned blocks of the compiled program (`regex_t`,
//! ops, operands, string pool, repeat ranges and optimizer string) are
//! allocated through the account. Parse temporaries, the name table, callout
//! data, the process-wide Oniguruma registries and every search-time
//! allocation stay on the legacy C allocator and are not charged.

use std::alloc::{alloc, dealloc, Layout};
use std::any::Any;
use std::cell::OnceCell;
use std::os::raw::{c_int, c_void};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe, RefUnwindSafe, UnwindSafe};
use std::ptr::null_mut;
use std::sync::Arc;
use std::{error, fmt, mem, process};

use crate::{EncodedChars, Error, Regex, RegexOptions, Syntax, REGEX_NEW_MUTEX};

/// Account charged for every retained compiled-backing block of an admitted
/// regex.
pub trait RegexAccount: Send + Sync + RefUnwindSafe + UnwindSafe + 'static {
    /// Typed refusal, returned unchanged.
    type Refusal: Send + 'static;

    /// Admits `bytes` before allocation; a refusal leaves the account unchanged.
    /// Must not compile Oniguruma regexes or call onig APIs (called under the compile lock).
    fn try_acquire(&self, bytes: usize) -> Result<(), Self::Refusal>;

    /// Returns bytes after their block was freed; any thread; must not panic.
    fn release(&self, bytes: usize);
}

/// Failure of [`Regex::new_with_account`]. Only the first failure of one
/// construction is reported.
#[derive(Debug)]
pub enum RegexAdmissionError<R> {
    /// The account refused a block.
    Refused(R),
    /// An admitted block could not be allocated; its charge was released.
    Allocation {
        /// Full size of the block, header included.
        bytes: usize,
    },
    /// A backing size computation overflows.
    CapacityOverflow,
    /// Any error of the legacy compilation, unchanged.
    Compile(Error),
}

impl<R> fmt::Display for RegexAdmissionError<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RegexAdmissionError::Refused(_) => f.write_str("Oniguruma regex backing was refused"),
            RegexAdmissionError::Allocation { bytes } => {
                write!(f, "Oniguruma regex allocation of {} bytes failed", bytes)
            }
            RegexAdmissionError::CapacityOverflow => {
                f.write_str("Oniguruma regex backing size overflows")
            }
            RegexAdmissionError::Compile(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl<R: error::Error + 'static> error::Error for RegexAdmissionError<R> {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            RegexAdmissionError::Refused(refusal) => Some(refusal),
            _ => None,
        }
    }
}

/// A failure recorded by a callback, outside the native error code.
enum Failure<R> {
    Refused(R),
    Allocation { bytes: usize },
    CapacityOverflow,
    Panicked(Box<dyn Any + Send>),
}

/// The per-construction failure slot; it lives on the constructor's stack and
/// is reached from callbacks only during `onig_basal_new`, on this thread.
struct Attempt<R> {
    first: OnceCell<Failure<R>>,
}

impl<R> Attempt<R> {
    /// Keeps the first failure. Dropping a later one must not unwind into C.
    fn record(&self, failure: Failure<R>) {
        if let Err(later) = self.first.set(failure) {
            abort_on_panic(catch_unwind(AssertUnwindSafe(move || drop(later))));
        }
    }
}

/// A panic here would leave the account or a C caller inconsistent. The
/// payload is leaked because dropping it could panic again.
fn abort_on_panic(result: std::thread::Result<()>) {
    if let Err(payload) = result {
        mem::forget(payload);
        process::abort();
    }
}

fn release_or_abort<A: RegexAccount>(account: &A, bytes: usize) {
    abort_on_panic(catch_unwind(|| account.release(bytes)));
}

fn admit<A: RegexAccount>(
    account: &A,
    bytes: usize,
    align: usize,
) -> Result<*mut c_void, Failure<A::Refusal>> {
    let layout = Layout::from_size_align(bytes, align).map_err(|_| Failure::CapacityOverflow)?;
    if layout.size() == 0 {
        // Never requested by onig_basal.c: every block carries a header.
        return Err(Failure::CapacityOverflow);
    }
    account.try_acquire(bytes).map_err(Failure::Refused)?;
    // SAFETY: `layout` is valid and nonzero-sized. The block is returned to C,
    // which frees it only through `deallocate::<A>` with this size and align.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        release_or_abort(account, bytes);
        return Err(Failure::Allocation { bytes });
    }
    Ok(block.cast())
}

unsafe extern "C" fn allocate<A: RegexAccount>(
    owner: *mut c_void,
    attempt: *mut c_void,
    bytes: usize,
    align: usize,
) -> *mut c_void {
    // SAFETY: `owner` is `Arc::into_raw` of an `Arc<A>` that the constructor
    // keeps alive until C returns; C calls this only while `attempt` points at
    // the constructor's live `Attempt<A::Refusal>` on this thread.
    let (account, attempt) =
        unsafe { (&*owner.cast::<A>(), &*attempt.cast::<Attempt<A::Refusal>>()) };
    match catch_unwind(|| admit(account, bytes, align)) {
        Ok(Ok(block)) => block,
        Ok(Err(failure)) => {
            attempt.record(failure);
            null_mut()
        }
        Err(payload) => {
            attempt.record(Failure::Panicked(payload));
            null_mut()
        }
    }
}

unsafe extern "C" fn deallocate<A: RegexAccount>(
    owner: *mut c_void,
    block: *mut c_void,
    bytes: usize,
    align: usize,
) {
    // SAFETY: `owner` is the live `Arc<A>` pointer held by the constructor or
    // by the regex until `drop_owner`, which C calls only after the last block.
    let account = unsafe { &*owner.cast::<A>() };
    // SAFETY: `block` came from `allocate::<A>` with exactly this size, read
    // back from its header, and the same constant align; C frees it once.
    unsafe {
        dealloc(
            block.cast(),
            Layout::from_size_align_unchecked(bytes, align),
        )
    };
    release_or_abort(account, bytes);
}

unsafe extern "C" fn overflow<A: RegexAccount>(attempt: *mut c_void) {
    // SAFETY: C calls this only during `onig_basal_new`, while `attempt` points
    // at the constructor's live `Attempt<A::Refusal>` on this thread.
    let attempt = unsafe { &*attempt.cast::<Attempt<A::Refusal>>() };
    attempt.record(Failure::CapacityOverflow);
}

unsafe extern "C" fn drop_owner<A: RegexAccount>(owner: *mut c_void) {
    // SAFETY: a successful construction moved this `Arc::into_raw` owner into
    // the regex; `onig_free` calls this exactly once, after its last block.
    let owner = unsafe { Arc::from_raw(owner.cast_const().cast::<A>()) };
    abort_on_panic(catch_unwind(move || drop(owner)));
}

impl Regex {
    /// Compiles `pattern` exactly like [`Regex::new`], but every retained
    /// compiled-backing block is admitted on `account` before its allocation
    /// and released after it is freed, which is when this regex is dropped.
    ///
    /// The first refusal, allocation failure, size overflow or panic of the
    /// account is returned; a panic resumes after the compile lock and the
    /// account are released. No callback unwinds through C. Other errors are
    /// the legacy compilation errors.
    pub fn new_with_account<A: RegexAccount>(
        pattern: &str,
        account: Arc<A>,
    ) -> Result<Regex, RegexAdmissionError<A::Refusal>> {
        let syntax = Syntax::default();
        let attempt = Attempt {
            first: OnceCell::new(),
        };
        let mut reg: onig_sys::OnigRegex = null_mut();
        let mut error = onig_sys::OnigErrorInfo {
            enc: null_mut(),
            par: null_mut(),
            par_end: null_mut(),
        };

        let (code, owner) = {
            // Same lock as `onig_new`; no callback unwinds, so it is never poisoned here.
            let _guard = REGEX_NEW_MUTEX.lock().unwrap();
            let owner = Arc::into_raw(account);
            let basal = onig_sys::OnigBasalAccount {
                owner: owner.cast_mut().cast(),
                attempt: (&attempt as *const Attempt<A::Refusal>).cast_mut().cast(),
                allocate: Some(allocate::<A>),
                deallocate: Some(deallocate::<A>),
                overflow: Some(overflow::<A>),
                drop_owner: Some(drop_owner::<A>),
            };
            // SAFETY: the pattern bounds and syntax are those `Regex::new`
            // passes; `reg`, `error`, `basal` and `attempt` outlive the call.
            // The trampolines match `A` and never unwind.
            let code = unsafe {
                onig_sys::onig_basal_new(
                    &mut reg,
                    pattern.start_ptr(),
                    pattern.limit_ptr(),
                    RegexOptions::REGEX_OPTION_NONE.bits(),
                    pattern.encoding(),
                    syntax as *const Syntax as *mut Syntax as *mut onig_sys::OnigSyntaxType,
                    &mut error,
                    &basal,
                )
            };
            (code, owner)
        };
        let failure = attempt.first.into_inner();

        if code == onig_sys::ONIG_NORMAL as c_int {
            let regex = Regex { raw: reg };
            return match failure {
                None => Ok(regex),
                Some(failure) => {
                    // Dropping releases every block and the owner.
                    drop(regex);
                    Err(failure.into_error())
                }
            };
        }

        // SAFETY: on failure C released every block and never called
        // `drop_owner`, so this still owns the `Arc::into_raw` reference.
        drop(unsafe { Arc::from_raw(owner) });
        match failure {
            Some(failure) => Err(failure.into_error()),
            None => Err(RegexAdmissionError::Compile(Error::from_code_and_info(
                code, &error,
            ))),
        }
    }
}

impl<R> Failure<R> {
    /// The typed error; a panic resumes here, after the lock and owner drop.
    fn into_error(self) -> RegexAdmissionError<R> {
        match self {
            Failure::Refused(refusal) => RegexAdmissionError::Refused(refusal),
            Failure::Allocation { bytes } => RegexAdmissionError::Allocation { bytes },
            Failure::CapacityOverflow => RegexAdmissionError::CapacityOverflow,
            Failure::Panicked(payload) => resume_unwind(payload),
        }
    }
}
