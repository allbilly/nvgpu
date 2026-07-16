// ponytail: minimal OpenCL vector add, targets first available platform/device.
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

static void clcheck(cl_int e, const char *what) {
    if (e != CL_SUCCESS) { fprintf(stderr, "%s failed: %d\n", what, e); exit(1); }
}

static const char *src =
"__kernel void add(__global const float *a, __global const float *b,\n"
"                  __global float *c, const unsigned int n) {\n"
"  size_t i = get_global_id(0);\n"
"  if (i < n) c[i] = a[i] + b[i];\n"
"}\n";

int main(void) {
    cl_platform_id plat;
    cl_device_id dev;
    clcheck(clGetPlatformIDs(1, &plat, NULL), "get platform");
    clcheck(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL), "get device");

    char name[128] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("device: %s\n", name);

    cl_int create_err = CL_SUCCESS;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &create_err);
    if (!ctx) { fprintf(stderr, "create context failed: %d\n", create_err); return 1; }
    /* Kepler's Rusticl/Nouveau path is OpenCL 1.2-era.  Use the legacy queue
       constructor so it does not enter the OpenCL-3 properties path. */
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &create_err);
    if (!q) { fprintf(stderr, "create queue failed: %d\n", create_err); return 1; }

    const int N = 1024;
    float *a = malloc(N * sizeof(float)), *b = malloc(N * sizeof(float)), *c = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(i * 2); }

    cl_mem ma = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(float), a, NULL);
    cl_mem mb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(float), b, NULL);
    cl_mem mc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N*sizeof(float), NULL, NULL);

    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, NULL);
    clcheck(clBuildProgram(prog, 1, &dev, NULL, NULL, NULL), "build");
    cl_kernel kern = clCreateKernel(prog, "add", NULL);

    clcheck(clSetKernelArg(kern, 0, sizeof(cl_mem), &ma), "arg0");
    clcheck(clSetKernelArg(kern, 1, sizeof(cl_mem), &mb), "arg1");
    clcheck(clSetKernelArg(kern, 2, sizeof(cl_mem), &mc), "arg2");
    clcheck(clSetKernelArg(kern, 3, sizeof(unsigned int), &N), "arg3");

    size_t global = N;
    clcheck(clEnqueueNDRangeKernel(q, kern, 1, NULL, &global, NULL, 0, NULL, NULL), "enqueue");
    clcheck(clEnqueueReadBuffer(q, mc, CL_TRUE, 0, N*sizeof(float), c, 0, NULL, NULL), "read");

    int ok = 1;
    for (int i = 0; i < N; i++) if (c[i] != (float)(i * 3)) { ok = 0; break; }
    printf("first 5: %.0f %.0f %.0f %.0f %.0f\n", c[0], c[1], c[2], c[3], c[4]);
    printf("%s\n", ok ? "PASS" : "FAIL");

    clReleaseMemObject(ma); clReleaseMemObject(mb); clReleaseMemObject(mc);
    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    free(a); free(b); free(c);
    return ok ? 0 : 1;
}
