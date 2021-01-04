// SPDX-License-Identifier: MIT
#include <cmath>
#include <cstring>

#include <gtest/gtest.h>

#include "vcuda.h"

#define ASSERT_CUDA_SUCCESS(f) ASSERT_EQ(CUDA_SUCCESS, f)

__global__ (static void kernel)(int n, int *a, int *b, int *c) {
  __global_init__;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    c[idx] = a[idx] + b[idx];
}

TEST(Apptest, VecAdd) {
  int n = 1023;
  int *ha, *hb, *hc;
  int *da, *db, *dc;

  ha = new int[n];
  hb = new int[n];
  hc = new int[n];

  ASSERT_CUDA_SUCCESS(cudaMalloc((void**)&da, n * sizeof(*da)));
  ASSERT_CUDA_SUCCESS(cudaMalloc((void**)&db, n * sizeof(*db)));
  ASSERT_CUDA_SUCCESS(cudaMalloc((void**)&dc, n * sizeof(*dc)));

  for (int i = 0; i < n; i++)
    ha[i] = rand(), hb[i] = rand();

  ASSERT_CUDA_SUCCESS(cudaMemcpy(da, ha, n * sizeof(*ha), cudaMemcpyHostToDevice));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(db, hb, n * sizeof(*hb), cudaMemcpyHostToDevice));

  const unsigned int nthrd = 256;
  const unsigned int nblk  = ceil(n / static_cast<double>(nthrd));

  /*--------------------------------------------------------------------------*/
  /* !! The following uses the /explicit/ syntax for launching kernels. !! */
  /*--------------------------------------------------------------------------*/
  void *kernelParams[] = { (void*)&n, (void*)&da, (void*)&db, (void*)&dc };

  ASSERT_CUDA_SUCCESS(cuLaunchKernel(
    CUfunction(VCUDA_kernel_kernel,
               { sizeof(n), sizeof(da), sizeof(db), sizeof(dc) }),
    nblk, 1, 1, nthrd, 1, 1, 0, 0, kernelParams, NULL));

  ASSERT_CUDA_SUCCESS(cudaMemcpy(hc, dc, n * sizeof(*dc), cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; i++)
    ASSERT_EQ(hc[i], ha[i] + hb[i]);

  ASSERT_CUDA_SUCCESS(cudaFree(da));
  ASSERT_CUDA_SUCCESS(cudaFree(db));
  ASSERT_CUDA_SUCCESS(cudaFree(dc));

  delete [] ha;
  delete [] hb;
  delete [] hc;
}
