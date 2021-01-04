// SPDX-License-Identifier: MIT
#include "vcuda/core/nullstream.h"
#include "vcuda/runtime.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
#ifdef VCUDA_WITH_LOGGING
vcuda::runtime::Runtime runtime;
#else
static vcuda::core::NullStream ns;

vcuda::runtime::Runtime runtime(ns);
#endif
