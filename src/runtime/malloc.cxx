// SPDX-License-Identifier: MIT
#include <cstddef>

#include "vcuda/core.h"
#include "vcuda/driver.h"
#include "vcuda/runtime.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
CUresult
vcuda::runtime::Runtime::malloc(void **dptr, std::size_t size) {
  return cuMemAlloc(dptr, size);
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
VCUDA_RUNTIME_EXPORT CUresult
cudaMalloc(void **dptr, std::size_t size) {
  return runtime.malloc(dptr, size);
}
