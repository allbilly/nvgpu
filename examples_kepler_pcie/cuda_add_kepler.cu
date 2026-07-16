// Minimal CUDA vector add for sm_30 (GTX 660 Ti). Build with CUDA 11.x:
// nvcc -arch=sm_30 -o cuda_add_kepler cuda_add_kepler.cu
#include <cstdio>
#include <cuda_runtime.h>
#define N 64
#define CHECK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return 1;} } while(0)
__global__ void add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  int n = 0; CHECK(cudaGetDeviceCount(&n));
  if (!n) { fprintf(stderr, "no CUDA device\n"); return 1; }
  cudaDeviceProp p{}; CHECK(cudaGetDeviceProperties(&p, 0));
  printf("device: %s cc=%d.%d\n", p.name, p.major, p.minor);
  float *ha = new float[N], *hb = new float[N], *hc = new float[N];
  float *da, *db, *dc;
  for (int i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = (float)(i * 2); hc[i] = -1; }
  CHECK(cudaMalloc(&da, N * sizeof(float)));
  CHECK(cudaMalloc(&db, N * sizeof(float)));
  CHECK(cudaMalloc(&dc, N * sizeof(float)));
  CHECK(cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice));
  add<<<(N + 31) / 32, 32>>>(da, db, dc, N);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(hc, dc, N * sizeof(float), cudaMemcpyDeviceToHost));
  int ok = 1;
  for (int i = 0; i < N; i++) if (hc[i] != (float)(i * 3)) ok = 0;
  printf("first 5: %.0f %.0f %.0f %.0f %.0f\n", hc[0], hc[1], hc[2], hc[3], hc[4]);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
