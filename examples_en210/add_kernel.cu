// add_kernel.cu — vector add for Tesla sm_12 (GT218 / GeForce 210)
// Computes: out[i] = a[i] + b[i]  for i = 0..N-1
extern "C" __global__ void add(const float *a, const float *b, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = a[i] + b[i];
}
