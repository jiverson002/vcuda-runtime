// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include "vcuda.h"

#define ASSERT_CUDA_SUCCESS(f) ASSERT_EQ(CUDA_SUCCESS, f)

TEST(Memtest, MallocFree) {
  char *ptr = NULL;

  ASSERT_CUDA_SUCCESS(cudaMalloc((void**)&ptr, 10));

  ASSERT_CUDA_SUCCESS(cudaFree(ptr));
}

TEST(Memtest, Memset) {
  int n = 1023, v = rand();
  int *h, *d;

  ASSERT_TRUE(NULL != (h = new int[n]));

  ASSERT_CUDA_SUCCESS(cudaMalloc((void**)&d, n * sizeof(*d)));

  ASSERT_CUDA_SUCCESS(cudaMemset(d, v, n * sizeof(*d)));

  ASSERT_CUDA_SUCCESS(cudaMemcpy(h, d, n * sizeof(*d), cudaMemcpyDeviceToHost));

  // construct correct value
  memset(&v, v, sizeof(v));

  for (int i = 0; i < n; i++)
    ASSERT_EQ(h[i], v);

  delete [] h;

  ASSERT_CUDA_SUCCESS(cudaFree(d));
}

TEST(Memtest, Memcpy) {
  int n = 1023;
  int *h1, *h2, *d;

  ASSERT_TRUE(NULL != (h1 = new int[n]));
  ASSERT_TRUE(NULL != (h2 = new int[n]));

  ASSERT_CUDA_SUCCESS(cudaMalloc((void**)&d, n * sizeof(*d)));

  for (int i = 0; i < n; i++)
    h1[i] = rand();

  ASSERT_CUDA_SUCCESS(cudaMemcpy(d, h1, n * sizeof(*h1), cudaMemcpyHostToDevice));

  ASSERT_CUDA_SUCCESS(cudaMemcpy(h2, d, n * sizeof(*d), cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; i++)
    ASSERT_EQ(h1[i], h2[i]);

  delete [] h1;
  delete [] h2;

  ASSERT_CUDA_SUCCESS(cudaFree(d));
}
