// SPDX-License-Identifier: MIT
#include "vcuda/core.h"
#include "vcuda/driver.h"
#include "vcuda/runtime.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
CUresult
vcuda::runtime::Runtime::memset(void *dst, const int value, std::size_t count) {
  return cuMemset(dst, value, count);
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
VCUDA_RUNTIME_EXPORT CUresult
cudaMemset(void *dst, const int value, std::size_t count) {
  return runtime.memset(dst, value, count);
}
