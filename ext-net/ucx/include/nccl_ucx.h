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
static const ucp_tag_t tag_mask = 0xFFFFFFFFFFFFFFFF; //recieve any message

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

ncclResult_t ucx_ptr_support(int dev, int* supported_types) {
  *supported_types = (NCCL_PTR_HOST | NCCL_PTR_CUDA);
  //*supported_types = NCCL_PTR_HOST;
  return ncclSuccess;
}

#define REG_ALIGN (4096)
ncclResult_t ucx_regmr(void* comm, void* data, int size, int type, void** mhandle){
  ucp_mem_map_params_t mmap_params;
  ucp_context_h *ctx = (ucp_context_h*)comm;

  // uint64_t addr = (uint64_t)data;
  // uint64_t reg_addr = addr & (~(REG_ALIGN-1));
  // uint64_t reg_size = addr+size - reg_addr;
  // reg_size = ((reg_size + REG_ALIGN-1) / REG_ALIGN ) * REG_ALIGN;

  uint64_t addr = (uint64_t)data;
  uint64_t reg_addr = addr;
  uint64_t reg_size = size;
  reg_size = ((reg_size + REG_ALIGN-1) / REG_ALIGN ) * REG_ALIGN;


  //fprintf(stderr, "want to map addr %p %zu memtype %d\n", (void*)reg_addr, reg_size, type);
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH; 
  mmap_params.address    = (void*)reg_addr;
  mmap_params.length     = reg_size;
  ucp_mem_map(*ctx, &mmap_params, (ucp_mem_h*)mhandle);
  return ncclSuccess;
}

ncclResult_t ucx_deregmr(void* comm, void* mhandle){
  ucp_context_h *ctx = (ucp_context_h*)comm;
  ucp_mem_unmap(*ctx, mhandle);
  return ncclSuccess;
}

ncclResult_t ucx_flush(void* recv_comm, void* data, int size, void* mhandle) {
  return ncclSuccess;
}

#endif