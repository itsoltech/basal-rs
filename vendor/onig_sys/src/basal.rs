//! basal-rs addition: hand-written bindings for `oniguruma/src/onig_basal.h`,
//! which is not part of `oniguruma.h` or the generated bindings.

use std::os::raw::{c_int, c_void};

use crate::{OnigEncoding, OnigErrorInfo, OnigOptionType, OnigRegex, OnigSyntaxType, OnigUChar};

/// Mirrors `OnigBasalAccount`. `onig_basal_new` copies it into the regex.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OnigBasalAccount {
    /// Opaque retained owner, transferred to the regex on success only.
    pub owner: *mut c_void,
    /// Per-construction failure slot; cleared before success returns.
    pub attempt: *mut c_void,
    /// Charges and allocates one whole block, or records a failure and returns null.
    pub allocate: Option<
        unsafe extern "C" fn(
            owner: *mut c_void,
            attempt: *mut c_void,
            bytes: usize,
            align: usize,
        ) -> *mut c_void,
    >,
    /// Frees one whole block, then releases its charge.
    pub deallocate: Option<
        unsafe extern "C" fn(owner: *mut c_void, block: *mut c_void, bytes: usize, align: usize),
    >,
    /// Records a backing-size overflow.
    pub overflow: Option<unsafe extern "C" fn(attempt: *mut c_void)>,
    /// Drops the owner after the regex released every block.
    pub drop_owner: Option<unsafe extern "C" fn(owner: *mut c_void)>,
}

extern "C" {
    /// `onig_new` whose retained compiled backing is charged to `account`.
    pub fn onig_basal_new(
        reg: *mut OnigRegex,
        pattern: *const OnigUChar,
        pattern_end: *const OnigUChar,
        option: OnigOptionType,
        enc: OnigEncoding,
        syntax: *mut OnigSyntaxType,
        einfo: *mut OnigErrorInfo,
        account: *const OnigBasalAccount,
    ) -> c_int;
}
