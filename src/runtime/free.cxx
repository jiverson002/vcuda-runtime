// SPDX-License-Identifier: MIT
#include "vcuda/core.h"
#include "vcuda/driver.h"
#include "vcuda/runtime.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
CUresult
vcuda::runtime::Runtime::free(void *dptr) {
  return cuMemFree(dptr);
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
VCUDA_RUNTIME_EXPORT CUresult
cudaFree(void *dptr) {
  return runtime.free(dptr);
}
