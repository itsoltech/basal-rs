#ifndef ONIG_BASAL_H
#define ONIG_BASAL_H
/**********************************************************************
  onig_basal.h - basal-rs addition to Oniguruma (BSD-2-Clause, see COPYING)

  Opt-in constructor whose regex charges its retained compiled backing
  (regex_t, ops/ocs, operands, string pool, repeat ranges and optimizer
  exact string) to a caller account until the regex is freed. It is not
  part of oniguruma.h or the generated bindings.
**********************************************************************/
#include <stddef.h>
#include "oniguruma.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OnigBasalAccount {
  void* owner;    /* opaque retained owner; transferred to the regex on success */
  void* attempt;  /* per-construction failure slot; NULL outside onig_basal_new */
  void* (*allocate)(void* owner, void* attempt, size_t bytes, size_t align);
  void  (*deallocate)(void* owner, void* block, size_t bytes, size_t align);
  void  (*overflow)(void* attempt);
  void  (*drop_owner)(void* owner);
} OnigBasalAccount;

/* Same compilation as onig_new() with ONIGENC_CASE_FOLD_DEFAULT. A valid reg
   always receives NULL on failure. On success the regex owns account->owner
   and onig_free() hands it to drop_owner after every block is released. On
   failure every admitted block is released and drop_owner is not called. */
ONIG_EXTERN int onig_basal_new(OnigRegex* reg, const OnigUChar* pattern,
  const OnigUChar* pattern_end, OnigOptionType option, OnigEncoding enc,
  OnigSyntaxType* syntax, OnigErrorInfo* einfo, const OnigBasalAccount* account);

#ifdef __cplusplus
}
#endif

#endif /* ONIG_BASAL_H */
