/*************************************************************************
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_UCX_H_
#define NCCL_UCX_H_

#include "nccl_ucx_log.h"
#include "nccl.h"
#include <ucp/api/ucp.h>
#include "ibvwrap.h"

#define UCXCHECK(cmd) do {                               \
  int e = cmd;                                           \
  if( UCS_OK != e ) {                                    \
    NCCL_UCX_WARN("Failed: UCX error %s:%d '%d' %s\n",   \
        __FILE__,__LINE__, e, ucs_status_string(e));     \
    return ncclInternalError;                            \
  }                                                      \
} while(0)

#define UCXCHECK_VOID(cmd) do {                               \
  int e = cmd;                                           \
  if( UCS_OK != e ) {                                    \
    NCCL_UCX_WARN("Failed: UCX error %s:%d '%d' %s\n",   \
        __FILE__,__LINE__, e, ucs_status_string(e));     \
  }                                                      \
} while(0)

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);


ncclDebugLogger_t ucx_log_function = NULL;
static const ucp_tag_t tag  = 0xABADBABE;
static const ucp_tag_t tag_mask = 0xFFFFFFFFFFFFFFFF;

static int ncclNIbDevs = -1;

#define MAXNAMESIZE 64
struct ncclIbDev {
  int device;
  uint8_t port;
  uint8_t link;
  struct ibv_context* context;
  char devName[MAXNAMESIZE];
  struct sockaddr_storage addr;
};

struct userIbDev {
  char devName[MAXNAMESIZE];
  uint16_t port_en;
};

struct ucx_mhandle {
  ucp_mem_h ucp_memh;
  ucp_rkey_h rkey;
};
typedef struct ucx_mhandle ucx_mhandle;

#define MAX_IB_DEVS 16
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
struct userIbDev userIbDevs[MAX_IB_DEVS];

ncclResult_t ucx_devices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t ucx_pci_path(int dev, char** path) {
  char devicepath[PATH_MAX];
  snprintf(devicepath, PATH_MAX, "/sys/class/infiniband/%s/device", ncclIbDevs[dev].devName);
  *path = realpath(devicepath, NULL);
  if (*path == NULL) {
    NCCL_UCX_WARN("Could not find real path of %s", devicepath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclIbGdrSupport(int ibDev) {
  static int moduleLoaded = -1;
  if (moduleLoaded == -1) {
    moduleLoaded = (access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == -1) ? 0 : 1;
  }
  if (moduleLoaded == 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t ucx_ptr_support(int dev, int* supported_types) {
  *supported_types = (NCCL_PTR_HOST | NCCL_PTR_CUDA);
  //*supported_types = NCCL_PTR_HOST;
  return ncclSuccess;
}


#endif
