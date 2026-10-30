// Modified for basal-rs: optionally share a regex whose compiled backing is admitted.
use crate::tokenizer::pattern::Pattern;
use crate::utils::merge_storage::{AdmittedRegexError, RegexCharge, TableRefusal};
use crate::{Offsets, Result};
use onig::{Regex, RegexAdmissionError};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

pub struct SysRegex {
    regex: Backing,
}

/// Legacy regexes are owned. Admitted regexes are shared by clones, so their
/// charged backing lives until the last owner drops it.
enum Backing {
    Owned(Regex),
    Shared(Arc<Regex>),
}

impl SysRegex {
    fn regex(&self) -> &Regex {
        match &self.regex {
            Backing::Owned(regex) => regex,
            Backing::Shared(regex) => regex,
        }
    }

    pub fn find_iter<'r, 't>(&'r self, inside: &'t str) -> onig::FindMatches<'r, 't> {
        self.regex().find_iter(inside)
    }

    pub fn new(
        regex_str: &str,
    ) -> std::result::Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        Ok(Self {
            regex: Backing::Owned(Regex::new(regex_str)?),
        })
    }

    /// Compiles like [`SysRegex::new`], admitting the retained compiled
    /// backing on `account`. Refusals stay typed; compile errors are boxed.
    pub(crate) fn new_admitted(
        regex_str: &str,
        account: &Arc<RegexCharge>,
    ) -> std::result::Result<Self, AdmittedRegexError> {
        match Regex::new_with_account(regex_str, Arc::clone(account)) {
            Ok(regex) => Ok(Self {
                regex: Backing::Shared(Arc::new(regex)),
            }),
            Err(RegexAdmissionError::Refused(refusal)) => Err(AdmittedRegexError::Refused(refusal)),
            Err(RegexAdmissionError::Allocation { bytes }) => {
                Err(AdmittedRegexError::Refused(TableRefusal::Allocation {
                    bytes,
                }))
            }
            Err(RegexAdmissionError::CapacityOverflow) => {
                Err(AdmittedRegexError::Refused(TableRefusal::CapacityOverflow))
            }
            Err(RegexAdmissionError::Compile(error)) => {
                Err(AdmittedRegexError::Backend(Box::new(error)))
            }
        }
    }

    /// Another owner of an admitted regex's backing; `None` for a legacy regex.
    pub(crate) fn share(&self) -> Option<Self> {
        match &self.regex {
            Backing::Owned(_) => None,
            Backing::Shared(regex) => Some(Self {
                regex: Backing::Shared(Arc::clone(regex)),
            }),
        }
    }
}

/// The derived output of the legacy single-field struct.
impl fmt::Debug for SysRegex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SysRegex")
            .field("regex", self.regex())
            .finish()
    }
}

impl Pattern for &Regex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        for (start, end) in self.find_iter(inside) {
            if prev != start {
                splits.push(((prev, start), false));
            }
            splits.push(((start, end), true));
            prev = end;
        }
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false))
        }
        Ok(splits)
    }
}
