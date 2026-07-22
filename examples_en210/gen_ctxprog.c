// Standalone generator for NV50 ctxprog + ctxvals.
// Compiles ctxnv50.c with stubs for kernel functions.
// Output: two binary files (ctxprog.bin, ctxvals.bin)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// --- Stubs for kernel types ---
struct nvkm_gpuobj {
  uint8_t *data;
  uint32_t size;
};

struct nvkm_ram {
  int type;
};

struct nvkm_fb {
  struct nvkm_ram *ram;
};

struct nvkm_device {
  int chipset;
  struct nvkm_fb *fb;
};

// --- Stubs for kernel functions ---
#define BUG_ON(x) do { if (x) { fprintf(stderr, "BUG_ON at %s:%d\n", __FILE__, __LINE__); exit(1); } } while(0)

static inline uint32_t nvkm_rd32(struct nvkm_device *dev, uint32_t reg) {
  // Return the units bitmap for 0x1540
  // GT218: 1 TPC (bit 0), 2 MPs (bits 24-25), plus some high bits
  if (reg == 0x001540) return 0xf3010001;
  fprintf(stderr, "nvkm_rd32: unhandled reg 0x%08x\n", reg);
  return 0;
}

static inline void nvkm_wr32(struct nvkm_device *dev, uint32_t reg, uint32_t val) {
  // No-op for generation
}

static inline void nvkm_wo32(struct nvkm_gpuobj *obj, uint32_t offset, uint32_t val) {
  if (offset + 4 > obj->size) {
    fprintf(stderr, "nvkm_wo32: overflow at offset 0x%x (size 0x%x)\n", offset, obj->size);
    return;
  }
  *(uint32_t*)(obj->data + offset) = val;
}

// --- Pull in ctxnv40.h definitions ---
// We need to define the struct and inline functions before ctxnv50.c
#define NVKM_RAM_TYPE_GDDR5 10

// Include ctxnv40.h inline functions (can't #include directly, so paste here)
#include "ctxnv40_defs.h"

// --- Now include ctxnv50.c ---
#include "ctxnv50_impl.c"

// --- Main ---
int main(int argc, char **argv) {
  int chipset = 0xa8; // GT218
  if (argc > 1) chipset = strtol(argv[1], NULL, 16);

  // Set up device
  struct nvkm_ram ram = { .type = 0 }; // not GDDR5
  struct nvkm_fb fb = { .ram = &ram };
  struct nvkm_device dev = { .chipset = chipset, .fb = &fb };

  // Phase 1: Generate ctxprog (PROG mode)
  uint32_t *ctxprog = calloc(512, 4);
  struct nvkm_grctx ctx = {
    .device = &dev,
    .mode = NVKM_GRCTX_PROG,
    .ucode = ctxprog,
    .data = NULL,
    .ctxprog_max = 512,
    .ctxprog_len = 0,
    .ctxprog_reg = 0,
    .ctxvals_pos = 0,
    .ctxvals_base = 0,
  };
  memset(ctx.ctxprog_label, 0, sizeof(ctx.ctxprog_label));

  nv50_grctx_generate(&ctx);
  uint32_t ctxprog_len = ctx.ctxprog_len;
  uint32_t ctxvals_size = ctx.ctxvals_pos * 4;

  fprintf(stderr, "ctxprog: %u instructions, ctxvals_size: 0x%x bytes\n",
          ctxprog_len, ctxvals_size);

  // Phase 2: Generate ctxvals (VALS mode)
  uint8_t *ctxvals = calloc(ctxvals_size + 0x1000, 1); // extra padding
  struct nvkm_gpuobj vals_obj = { .data = ctxvals, .size = ctxvals_size + 0x1000 };

  struct nvkm_grctx ctx2 = {
    .device = &dev,
    .mode = NVKM_GRCTX_VALS,
    .ucode = NULL,
    .data = &vals_obj,
    .ctxprog_max = 512,
    .ctxprog_len = 0,
    .ctxprog_reg = 0,
    .ctxvals_pos = 0,
    .ctxvals_base = 0,
  };
  memset(ctx2.ctxprog_label, 0, sizeof(ctx2.ctxprog_label));

  nv50_grctx_generate(&ctx2);

  fprintf(stderr, "ctxvals generated: 0x%x bytes\n", ctxvals_size);

  // Write output files
  FILE *f;
  f = fopen("ctxprog.bin", "wb");
  fwrite(ctxprog, 4, ctxprog_len, f);
  fclose(f);

  f = fopen("ctxvals.bin", "wb");
  fwrite(ctxvals, 1, ctxvals_size, f);
  fclose(f);

  // Also write as Python-importable format
  f = fopen("ctxprog.py", "w");
  fprintf(f, "# Auto-generated ctxprog for chipset 0x%02x\n", chipset);
  fprintf(f, "ctxprog = [\n");
  for (uint32_t i = 0; i < ctxprog_len; i++) {
    fprintf(f, "  0x%08x,\n", ctxprog[i]);
  }
  fprintf(f, "]\n");
  fprintf(f, "ctxvals_size = 0x%x\n", ctxvals_size);
  fclose(f);

  fprintf(stderr, "Wrote ctxprog.bin (%u bytes), ctxvals.bin (0x%x bytes), ctxprog.py\n",
          ctxprog_len * 4, ctxvals_size);

  free(ctxprog);
  free(ctxvals);
  return 0;
}
