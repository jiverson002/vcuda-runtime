// SPDX-License-Identifier: MIT
#include "vcuda/core.h"
#include "vcuda/driver.h"
#include "vcuda/runtime.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
CUresult
vcuda::runtime::Runtime::memcpy(void *dst, const void *src, std::size_t count,
                                enum cudaMemcpyKind kind)
{
  switch (kind) {
    case cudaMemcpyDeviceToHost:
    return cuMemcpyDtoH(dst, const_cast<void*>(src), count);
    break;

    case cudaMemcpyHostToDevice:
    return cuMemcpyHtoD(dst, src, count);
    break;

    default:
    return CUDA_ERROR_INVALID_MEMCPY_DIRECTION;
  }
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
VCUDA_RUNTIME_EXPORT CUresult
cudaMemcpy(void *dst, const void *src, std::size_t count,
           enum cudaMemcpyKind kind)
{
  return runtime.memcpy(dst, src, count, kind);
}
