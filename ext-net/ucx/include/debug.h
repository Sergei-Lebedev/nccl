/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

#include "nccl_net.h"

extern ncclDebugLogger_t ucx_log_function;

#define WARN(...) ucx_log_function(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ucx_log_function(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ucx_log_function(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#else
#define TRACE(...)
#endif

#endif
