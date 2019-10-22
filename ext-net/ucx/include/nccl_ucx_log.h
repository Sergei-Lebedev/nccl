/*************************************************************************
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_UCX_LOG_H_
#define NCCL_UCX_LOG_H_

#ifdef _cplusplus
extern "C" {
#endif

#define NCCL_UCX_WARN(fmt, ...)                                       \
  (*ucx_log_function)(NCCL_LOG_WARN, NCCL_ALL, __PRETTY_FUNCTION__,   \
      __LINE__, "NET/UCX " fmt, ##__VA_ARGS__)

#define NCCL_UCX_INFO(flags, fmt, ...)                                \
  (*ucx_log_function)(NCCL_LOG_INFO, flags,                           \
      __PRETTY_FUNCTION__, __LINE__, "NET/UCX " fmt,                  \
##__VA_ARGS__)

#define NCCL_UCX_TRACE(flags, fmt, ...)                               \
  (*ucx_log_function)(NCCL_LOG_TRACE, flags,                          \
      __PRETTY_FUNCTION__, __LINE__, "NET/UCX " fmt,                  \
##__VA_ARGS__)

#ifdef _cplusplus
}
#endif
#endif
