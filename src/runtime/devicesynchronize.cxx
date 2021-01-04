// SPDX-License-Identifier: MIT
#include "vcuda/core.h"
#include "vcuda/driver.h"
#include "vcuda/runtime.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
CUresult
vcuda::runtime::Runtime::deviceSynchronize(void) {
  return ::driver.deviceSynchronize();
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
VCUDA_RUNTIME_EXPORT CUresult
cudaDeviceSynchronize(void) {
  return runtime.deviceSynchronize();
}
