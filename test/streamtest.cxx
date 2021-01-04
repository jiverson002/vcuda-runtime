// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include "vcuda.h"

#define ASSERT_CUDA_SUCCESS(f) ASSERT_EQ(CUDA_SUCCESS, f)

TEST(Streamtest, CreateDestroy) {
  CUstream stream1, stream2, stream3;

  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream1));
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream2));
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream3));

  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream2));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream1));
}
