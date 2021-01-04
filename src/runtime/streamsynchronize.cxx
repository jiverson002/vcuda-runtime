// SPDX-License-Identifier: MIT
#include "vcuda/core.h"
#include "vcuda/driver.h"
#include "vcuda/runtime.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
CUresult
vcuda::runtime::Runtime::streamSynchronize(CUstream hstream) {
  return ::driver.streamSynchronize(hstream);
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
VCUDA_RUNTIME_EXPORT CUresult
cudaStreamSynchronize(CUstream hstream) {
  return runtime.streamSynchronize(hstream);
}
