// SPDX-License-Identifier: MIT
#ifndef VCUDA_RUNTIME_H
#define VCUDA_RUNTIME_H 1
#include <cstddef>
#include <iostream>
#include <ostream>

#include "vcuda/core.h"
#include "vcuda/runtime/export.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
#ifndef GOTO
# define GOTO(lbl) do {\
  std::fprintf(stderr, "runtime: failure in %s @ %s:%d\n", __FILE__, __func__,\
               __LINE__);\
  if (0 != errno)\
    std::fprintf(stderr, "  errno: %s\n", std::strerror(errno));\
  std::fflush(stderr);\
  goto lbl;\
} while (0)
#endif

namespace vcuda {
  namespace runtime {
    class Runtime {
      public:
        Runtime(std::ostream &log = std::cerr) : log(log) { }
        ~Runtime() = default;

        CUresult deviceSynchronize(void);
        CUresult free(void *);
        CUresult malloc(void **, std::size_t);
        CUresult memcpy(void *, const void *, std::size_t, enum cudaMemcpyKind);
        CUresult memset(void *, const int, std::size_t);
        CUresult streamCreate(CUstream *);
        CUresult streamDestroy(CUstream);
        CUresult streamSynchronize(CUstream);
        CUresult version(int *);

      private:
        std::ostream &log; /*!< ostream for logging */
    };
  }
}

/*----------------------------------------------------------------------------*/
/*! One instance of runtime per program invocation. */
/*----------------------------------------------------------------------------*/
VCUDA_RUNTIME_EXPORT extern vcuda::runtime::Runtime runtime;

/*----------------------------------------------------------------------------*/
/*! Runtime API. */
/*----------------------------------------------------------------------------*/
#ifdef __cplusplus
extern "C" {
#endif

VCUDA_RUNTIME_EXPORT CUresult cudaDeviceSynchronize(void);
VCUDA_RUNTIME_EXPORT CUresult cudaDriverGetVersion(int *);
VCUDA_RUNTIME_EXPORT CUresult cudaFree(void *dptr);
VCUDA_RUNTIME_EXPORT CUresult cudaMalloc(void **dptr, std::size_t size);
VCUDA_RUNTIME_EXPORT CUresult cudaMemcpy(void *dst, const void *src, std::size_t count,
                                 enum cudaMemcpyKind kind);
VCUDA_RUNTIME_EXPORT CUresult cudaMemset(void *dst, const int value, std::size_t count);
VCUDA_RUNTIME_EXPORT CUresult cudaRuntimeGetVersion(int *);
VCUDA_RUNTIME_EXPORT CUresult cudaStreamCreate(CUstream *pstream);
VCUDA_RUNTIME_EXPORT CUresult cudaStreamDestroy(CUstream hstream);
VCUDA_RUNTIME_EXPORT CUresult cudaStreamSynchronize(CUstream hstream);

#ifdef __cplusplus
}
#endif

#endif // VCUDA_RUNTIME_H
