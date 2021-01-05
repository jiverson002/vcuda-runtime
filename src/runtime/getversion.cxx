// SPDX-License-Identifier: MIT
#include "vcuda/core.h"
#include "vcuda/driver.h"
#include "vcuda/runtime.h"
#include "vcuda/runtime/config.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
CUresult
vcuda::runtime::Runtime::version(int *runtimeVersion) {
  if (NULL == runtimeVersion)
    return CUDA_ERROR_INVALID_VALUE;

  *runtimeVersion = 1000 * VCUDA_RUNTIME_VERSION_MAJOR +
                    10   * VCUDA_RUNTIME_VERSION_MINOR;

  return CUDA_SUCCESS;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
VCUDA_RUNTIME_EXPORT CUresult
cudaDriverGetVersion(int *driverVersion) {
  return ::driver.version(driverVersion);
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
VCUDA_RUNTIME_EXPORT CUresult
cudaRuntimeGetVersion(int *runtimeVersion) {
  return runtime.version(runtimeVersion);
}
