#ifndef __CTXNV40_DEFS_H__
#define __CTXNV40_DEFS_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// nvkm_grctx struct (from ctxnv40.h, without kernel deps)
enum {
  NVKM_GRCTX_PROG,
  NVKM_GRCTX_VALS
};

struct nvkm_grctx {
  struct nvkm_device *device;
  int mode; // NVKM_GRCTX_PROG or NVKM_GRCTX_VALS
  uint32_t *ucode;
  struct nvkm_gpuobj *data;
  uint32_t ctxprog_max;
  uint32_t ctxprog_len;
  uint32_t ctxprog_reg;
  int ctxprog_label[32];
  uint32_t ctxvals_pos;
  uint32_t ctxvals_base;
};

// Inline functions from ctxnv40.h
static inline void cp_out(struct nvkm_grctx *ctx, uint32_t inst) {
  uint32_t *ctxprog = ctx->ucode;
  if (ctx->mode != NVKM_GRCTX_PROG) return;
  if (ctx->ctxprog_len == ctx->ctxprog_max) {
    fprintf(stderr, "ctxprog overflow!\n");
    exit(1);
  }
  ctxprog[ctx->ctxprog_len++] = inst;
}

static inline void cp_lsr(struct nvkm_grctx *ctx, uint32_t val) {
  cp_out(ctx, 0x00200000 | val);
}

static inline void cp_ctx(struct nvkm_grctx *ctx, uint32_t reg, uint32_t length) {
  ctx->ctxprog_reg = (reg - 0x00400000) >> 2;
  ctx->ctxvals_base = ctx->ctxvals_pos;
  ctx->ctxvals_pos = ctx->ctxvals_base + length;
  if (length > (0x000f0000 >> 16)) {
    cp_lsr(ctx, length);
    length = 0;
  }
  cp_out(ctx, 0x00100000 | (length << 16) | ctx->ctxprog_reg);
}

static inline void cp_name(struct nvkm_grctx *ctx, int name) {
  uint32_t *ctxprog = ctx->ucode;
  int i;
  if (ctx->mode != NVKM_GRCTX_PROG) return;
  ctx->ctxprog_label[name] = ctx->ctxprog_len;
  for (i = 0; i < (int)ctx->ctxprog_len; i++) {
    if ((ctxprog[i] & 0xfff00000) != 0xff400000) continue;
    if ((ctxprog[i] & 0x0001ff00) != ((name) << 8)) continue;
    ctxprog[i] = (ctxprog[i] & 0x00ff00ff) | (ctx->ctxprog_len << 8);
  }
}

static inline void _cp_bra(struct nvkm_grctx *ctx, uint32_t mod, int flag, int state, int name) {
  int ip = 0;
  if (mod != 2) {
    ip = ctx->ctxprog_label[name] << 8;
    if (ip == 0) ip = 0xff000000 | (name << 8);
  }
  cp_out(ctx, 0x00400000 | (mod << 18) | ip | flag | (state ? 0 : 0x00000080));
}

static inline void _cp_wait(struct nvkm_grctx *ctx, int flag, int state) {
  cp_out(ctx, 0x00500000 | flag | (state ? 0x00000080 : 0));
}

static inline void _cp_set(struct nvkm_grctx *ctx, int flag, int state) {
  cp_out(ctx, 0x00700000 | flag | (state ? 0x00000080 : 0));
}

static inline void cp_pos(struct nvkm_grctx *ctx, int offset) {
  ctx->ctxvals_pos = offset;
  ctx->ctxvals_base = ctx->ctxvals_pos;
  cp_lsr(ctx, ctx->ctxvals_pos);
  cp_out(ctx, 0x00600006);
}

static inline void gr_def(struct nvkm_grctx *ctx, uint32_t reg, uint32_t val) {
  if (ctx->mode != NVKM_GRCTX_VALS) return;
  reg = (reg - 0x00400000) / 4;
  reg = (reg - ctx->ctxprog_reg) + ctx->ctxvals_base;
  nvkm_wo32(ctx->data, reg * 4, val);
}

#endif

// cp_bra/cp_set/cp_wait macros (from ctxnv40.h)
#define cp_bra(c, f, s, n) _cp_bra((c), 0, CP_FLAG_##f, CP_FLAG_##f##_##s, n)
#define cp_cal(c, f, s, n) _cp_bra((c), 1, CP_FLAG_##f, CP_FLAG_##f##_##s, n)
#define cp_ret(c, f, s) _cp_bra((c), 2, CP_FLAG_##f, CP_FLAG_##f##_##s, 0)
#define cp_wait(c, f, s) _cp_wait((c), CP_FLAG_##f, CP_FLAG_##f##_##s)
#define cp_set(c, f, s) _cp_set((c), CP_FLAG_##f, CP_FLAG_##f##_##s)

// Kernel type stubs
typedef uint32_t u32;
typedef uint64_t u64;
typedef int bool;
#define true 1
#define false 0
#define GFP_KERNEL 0
static inline void *kmalloc(size_t size, int flags) { return malloc(size); }
static inline void kfree(void *p) { free(p); }
#define kzalloc_obj(t) calloc(1, sizeof(t))
#define ilog2(x) (31 - __builtin_clz(x))
#define lower_32_bits(x) ((u32)(x))
#define upper_32_bits(x) ((u32)(((u64)(x) >> 32)))
#define max(a,b) ((a) > (b) ? (a) : (b))
#define ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))
