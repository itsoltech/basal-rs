/**********************************************************************
  onig_basal.c - basal-rs addition to Oniguruma (BSD-2-Clause, see COPYING)

  Admitted retained compiled backing. A regex built by onig_basal_new()
  carries reg->basal; only the regcomp.c retained-family sites dispatch on
  it. Every admitted block is one account allocation of header + payload,
  charged by the account before it exists and released by the account after
  it is gone. Legacy regexes keep reg->basal == NULL and the old allocator.
**********************************************************************/
#include "regint.h"
#include "onig_basal.h"

/* The regex is the first member: the container block's payload is the regex. */
typedef struct {
  regex_t          reg;
  OnigBasalAccount account;
} BasalRegex;

#define BASAL_SIZE_MAX      (~(size_t )0)
#define BASAL_HEADER_SIZE   sizeof(OnigBasalHeader)
#define BASAL_HEADER_ALIGN  _Alignof(OnigBasalHeader)

extern int
onig_basal_reg_overflow(const OnigBasalAccount* a)
{
  if (IS_NOT_NULL(a->attempt))
    a->overflow(a->attempt);

  return ONIGERR_MEMORY;
}

/* NULL without any call outside a construction (never reached after one). */
extern void*
onig_basal_reg_alloc(const OnigBasalAccount* a, size_t n)
{
  OnigBasalHeader* h;

  if (IS_NULL(a->attempt)) return NULL;

  if (n > BASAL_SIZE_MAX - BASAL_HEADER_SIZE) {
    (void )onig_basal_reg_overflow(a);
    return NULL;
  }

  h = (OnigBasalHeader* )a->allocate(a->owner, a->attempt,
                                     BASAL_HEADER_SIZE + n, BASAL_HEADER_ALIGN);
  if (IS_NULL(h)) return NULL;

  h->bytes = BASAL_HEADER_SIZE + n;
  return (void* )(h + 1);
}

extern void
onig_basal_reg_free(const OnigBasalAccount* a, void* p)
{
  OnigBasalHeader* h;

  if (IS_NULL(p)) return ;

  h = (OnigBasalHeader* )p - 1;
  a->deallocate(a->owner, (void* )h, h->bytes, BASAL_HEADER_ALIGN);
}

/* The new block is charged while the old one lives (old + new overlap);
   on failure the old block and its charge are untouched. */
extern void*
onig_basal_reg_realloc(const OnigBasalAccount* a, void* p, size_t n)
{
  void* q;
  size_t old;

  if (IS_NULL(p)) return onig_basal_reg_alloc(a, n);

  old = ((OnigBasalHeader* )p - 1)->bytes - BASAL_HEADER_SIZE;
  q = onig_basal_reg_alloc(a, n);
  if (IS_NULL(q)) return NULL;

  xmemcpy(q, p, old < n ? old : n);
  onig_basal_reg_free(a, p);
  return q;
}

extern void*
onig_basal_reg_alloc_array(const OnigBasalAccount* a, void* p, int count,
                           size_t elem)
{
  if (count < 0 || (size_t )count > BASAL_SIZE_MAX / elem) {
    (void )onig_basal_reg_overflow(a);
    return NULL;
  }

  return onig_basal_reg_realloc(a, p, (size_t )count * elem);
}

/* onigenc_strdup() on the admitted allocator, within its int domain. */
extern UChar*
onig_basal_reg_strdup(const OnigBasalAccount* a, OnigEncoding enc,
                      const UChar* s, const UChar* end)
{
  ptrdiff_t slen;
  int term_len, i;
  UChar* r;

  slen = end - s;
  term_len = ONIGENC_MBC_MINLEN(enc);
  if (slen < 0 || slen > (ptrdiff_t )(INT_MAX - term_len)) {
    (void )onig_basal_reg_overflow(a);
    return NULL;
  }

  r = (UChar* )onig_basal_reg_alloc(a, (size_t )slen + (size_t )term_len);
  CHECK_NULL_RETURN(r);
  xmemcpy(r, s, (size_t )slen);

  for (i = 0; i < term_len; i++)
    r[slen + i] = (UChar )0;

  return r;
}

/* Called by onig_free() for an admitted regex: every family block is
   released, then the container, then the owner, from a stack copy. */
extern void
onig_basal_free_regex(regex_t* reg)
{
  OnigBasalAccount account;

  account = *reg->basal;
  onig_free_body(reg);
  onig_basal_reg_free(&account, (void* )reg);
  account.drop_owner(account.owner);
}

extern int
onig_basal_new(OnigRegex* reg, const OnigUChar* pattern,
               const OnigUChar* pattern_end, OnigOptionType option,
               OnigEncoding enc, OnigSyntaxType* syntax, OnigErrorInfo* einfo,
               const OnigBasalAccount* account)
{
  int r;
  BasalRegex* c;

  if (IS_NULL(reg)) return ONIGERR_INVALID_ARGUMENT;
  *reg = NULL;

  if (IS_NULL(account) || IS_NULL(account->allocate) ||
      IS_NULL(account->deallocate) || IS_NULL(account->overflow) ||
      IS_NULL(account->drop_owner))
    return ONIGERR_INVALID_ARGUMENT;

  c = (BasalRegex* )onig_basal_reg_alloc(account, sizeof(*c));
  if (IS_NULL(c)) return ONIGERR_MEMORY;

  c->account = *account;

  /* Unchanged; its memset leaves c->reg.basal NULL and it allocates nothing
     through the regex. */
  r = onig_reg_init(&c->reg, option, ONIGENC_CASE_FOLD_DEFAULT, enc, syntax);
  if (r != 0) {
    onig_basal_reg_free(account, (void* )c);
    return r;
  }

  c->reg.basal = &c->account;
  r = onig_compile(&c->reg, pattern, pattern_end, einfo);
  if (r != 0) {
    onig_free_body(&c->reg);
    onig_basal_reg_free(account, (void* )c);
    return r;
  }

  c->account.attempt = NULL;
  *reg = &c->reg;
  return ONIG_NORMAL;
}
