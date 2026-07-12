extern "C" __global__ void E_4(const float* a, const float* b, float* out) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  out[i] = a[i] + b[i];
}
