
/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#include "nccl.h"
#include "nccl_net.h"
#include "core.h"
#include "socket.h"
#include "nccl_ucx.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>

pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;

struct ucx_listen_handle {
  union socketAddress connectAddr;
  ucp_tag_t tag;
};
typedef struct ucx_listen_handle ucx_listen_handle;

struct ucx_listen_comm {
  int fd;
  int dev;
  ucp_context_h ctx;
  ucp_worker_h worker;
  ucp_tag_t tag;
};
typedef struct ucx_listen_comm ucx_listen_comm;

struct connect_msg {
  size_t addr_len;
};
typedef struct connect_msg connect_msg;

struct ucx_request {
  int completed;
  int size;
  ucp_worker_h worker;
};
typedef struct ucx_request ucx_request;

struct nccl_ucx_worker {
  ucp_context_h ctx;
  ucp_worker_h worker;
  int count;
};
static struct nccl_ucx_worker workers[MAX_IB_DEVS];

struct ucx_gpu_flush {
  int enabled;
  int hostMem;
  ucp_ep_h flush_ep;
};

struct ucx_ctx {
  ucp_context_h ucp_ctx;
  struct ucx_gpu_flush gpuFlush;
};
typedef struct ucx_ctx ucx_ctx;

struct ucx_send_comm {
  ucp_context_h ctx;
  struct ucx_gpu_flush gpuFlush;
  ucp_worker_h worker;
  ucp_ep_h ep;
  ucp_tag_t tag;
  ucp_tag_t ctag;
  int fd;
  int ready;
  uint32_t fifo_head;
  uint32_t fifo_tail;
  ucp_mem_h fifo_memh;
};
typedef struct ucx_send_comm ucx_send_comm;

struct ucx_recv_comm {
  ucp_context_h ctx;
  struct ucx_gpu_flush gpuFlush;
  ucp_worker_h worker;
  ucp_ep_h ep;
  ucp_tag_t tag;
  ucp_tag_t ctag;
  int fd;
  int ready;
  uint64_t rem_tail_addr;
  uint32_t tail;
  ucp_rkey_h rkey;
  connect_msg *msg;
  ucx_request *connect_req;
};
typedef struct ucx_recv_comm ucx_recv_comm;

static void request_init(void *request) {
  struct ucx_request *req = (struct ucx_request *)request;
  req->completed = 0;
}

static void send_handler(void *request, ucs_status_t status) {
  struct ucx_request *req = (struct ucx_request *)request;
  req->completed = 1;
}

static void recv_handler(void *request, ucs_status_t status, ucp_tag_recv_info_t *info) {
  struct ucx_request *req = (struct ucx_request *)request;
  req->completed = 1;
}

static union socketAddress nccl_ucx_if_addr;
static char if_name[MAX_IF_NAME_SIZE];

static ncclResult_t get_socket_addr(union socketAddress *addr) {
  memcpy(addr, &nccl_ucx_if_addr, sizeof(*addr));
  return ncclSuccess;
}

static ncclResult_t ucx_init_context(ucp_context_h *ctx, int dev) {
  ucp_params_t ucp_params;
  ucp_config_t *config;
  char ucx_dev_name[PATH_MAX];

  snprintf(ucx_dev_name, PATH_MAX, "%s:%d", ncclIbDevs[dev].devName, ncclIbDevs[dev].port);
  UCXCHECK(ucp_config_read("NCCL", NULL, &config));
  UCXCHECK(ucp_config_modify(config, "NET_DEVICES", ucx_dev_name));
  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_REQUEST_SIZE |
                          UCP_PARAM_FIELD_REQUEST_INIT;
  ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA;
  ucp_params.request_size = sizeof(struct ucx_request);
  ucp_params.request_init = request_init;
  UCXCHECK(ucp_init(&ucp_params, config, ctx));
  ucp_config_release(config);
}

static ncclResult_t ucx_init_worker(ucp_context_h ctx, ucp_worker_h *worker) {
  ucp_worker_params_t worker_params;
  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  UCXCHECK(ucp_worker_create(ctx, &worker_params, worker));
}

static ncclResult_t ucx_worker_get_netaddress(ucp_worker_h worker, ucp_address_t **address, size_t *address_length) {
  ucp_worker_attr_t attr;
  attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS |
                    UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
  attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;
  UCXCHECK(ucp_worker_query(worker, &attr));
  *address = malloc(attr.address_length);
  memcpy(*address, attr.address, attr.address_length);
  *address_length = attr.address_length;
  free(attr.address);
  return ncclSuccess;
}

#define UCX_SHARED_WORKER
static ncclResult_t ucx_get_ctx_and_worker(int dev, ucp_context_h *ctx, ucp_worker_h *worker, ucp_tag_t *newtag) {
  pthread_mutex_lock(&ncclIbLock);
#ifdef UCX_SHARED_WORKER
  if (ncclNIbDevs < dev) {
    NCCL_UCX_WARN("Device index is too large");
    return ncclSystemError;
  }
  if (workers[dev].count == 0) {
    ucx_init_context(&workers[dev].ctx, dev);
    ucx_init_worker(workers[dev].ctx, &workers[dev].worker);
  }
  *ctx = workers[dev].ctx;
  *worker = workers[dev].worker;
  if (newtag != NULL) {
    *newtag = tag + workers[dev].count;
  }
  ucp_worker_progress(*worker);
  workers[dev].count++;
#else
  ucx_init_context(ctx, dev);
  ucx_init_worker(*ctx, worker);
  if (newtag != NULL) {
    *newtag = tag;
  }
#endif
  pthread_mutex_unlock(&ncclIbLock);
  return ncclSuccess;
}

ncclResult_t ucx_init(ncclDebugLogger_t logFunction) {
  ucx_log_function = logFunction;
  if (ncclNIbDevs == -1)
  {
    pthread_mutex_lock(&ncclIbLock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1)
    {
      ncclNIbDevs = 0;
      if (findInterfaces(if_name, &nccl_ucx_if_addr, MAX_IF_NAME_SIZE, 1) != 1)
      {
        NCCL_UCX_WARN("NET/UCX : No IP interface found.");
        return ncclInternalError;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device **devices;

      // Check if user defined which IB device:port to use
      char *userIbEnv = getenv("NCCL_IB_HCA");
      struct netIf userIfs[MAX_IB_DEVS];
      int searchNot = userIbEnv && userIbEnv[0] == '^';
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs))
        return ncclInternalError;

      for (int d = 0; d < nIbDevs; d++) {
        struct ibv_context *context;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          NCCL_UCX_WARN("NET/UCX : Unable to open device %s", devices[d]->name);
          continue;
        }
        int found = 0;
        struct ibv_device_attr devAttr;
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          NCCL_UCX_WARN("NET/UCX : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) {
            return ncclInternalError;
          }
          continue;
        }
        for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
          struct ibv_port_attr portAttr;
          if (ncclSuccess != wrap_ibv_query_port(context, port, &portAttr)) {
            NCCL_UCX_WARN("NET/UCX : Unable to query port %d", port);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE)
            continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET)
            continue;

          // check against user specified HCAs/ports
          if (!(matchIfList(devices[d]->name, port, userIfs, nUserIfs) ^ searchNot)) {
            continue;
          }
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].context = context;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          workers[ncclNIbDevs].count = 0;
          ncclNIbDevs++;
          found++;
          //pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, context);
        }
        if (found == 0 && ncclSuccess != wrap_ibv_close_device(context)) {
          return ncclInternalError;
        }
      }
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) {
        return ncclInternalError;
      };
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT | NCCL_NET, "NET/UCX : No device found.");
    }
    else {
      char line[1024];
      line[0] = '\0';
      for (int d = 0; d < ncclNIbDevs; d++) {
        snprintf(line + strlen(line), 1023 - strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
                 ncclIbDevs[d].port, ncclIbDevs[d].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
      }
      line[1023] = '\0';
      INFO(NCCL_INIT | NCCL_NET, "NET/UCX : Using%s ; OOB %s", line, if_name);
    }
    pthread_mutex_unlock(&ncclIbLock);
  }
  return ncclSuccess;
}

ncclResult_t ucx_listen(int dev, void *handle, void **listen_comm) {
  ucx_listen_handle *my_handle;
  ucx_listen_comm *comm;

  comm = malloc(sizeof(ucx_listen_comm));
  memset(comm, 0, sizeof(ucx_listen_comm));
  static_assert(sizeof(ucx_listen_handle) < NCCL_NET_HANDLE_MAXSIZE, "ucx listen handle size too large");
  my_handle = (ucx_listen_handle *)handle;
  comm->dev = dev;
  NCCLCHECK(get_socket_addr(&(my_handle->connectAddr)));
  NCCLCHECK(createListenSocket(&comm->fd, &my_handle->connectAddr));
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker, &comm->tag));
  my_handle->tag = comm->tag;
  *listen_comm = comm;
  return ncclSuccess;
}

ncclResult_t ucx_connect(int dev, void *handle, void **send_comm) {
  ucp_address_t *my_addr;
  size_t local_addr_len;
  size_t rkey_buf_size;
  void *rkey_buf;

  ucx_listen_handle *recv_handle = (ucx_listen_handle *)handle;
  ucx_send_comm *comm = (ucx_send_comm *) calloc(1, sizeof(ucx_send_comm));
 
  NCCLCHECK(connectAddress(&comm->fd, &recv_handle->connectAddr));
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker, &comm->ctag));
  comm->tag = recv_handle->tag;
  comm->gpuFlush.enabled = 0;
  NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));
  NCCL_UCX_INFO(NCCL_NET, "Worker address length: %zu", local_addr_len);

  ucp_mem_map_params_t mmap_params;
  uint64_t tail_adr = (uint64_t)&comm->fifo_tail;
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  mmap_params.address = (void*)tail_adr;
  mmap_params.length = sizeof(uint32_t);
  ucp_mem_map(comm->ctx, &mmap_params, &comm->fifo_memh);
  ucp_rkey_pack(comm->ctx, comm->fifo_memh, &rkey_buf, &rkey_buf_size);

  NCCLCHECK(socketSend(comm->fd, &rkey_buf_size, sizeof(size_t)));
  NCCLCHECK(socketSend(comm->fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketSend(comm->fd, &tail_adr, sizeof(uint64_t)));
  NCCLCHECK(socketSend(comm->fd, &local_addr_len, sizeof(size_t)));
  NCCLCHECK(socketSend(comm->fd, my_addr, local_addr_len));
  NCCLCHECK(socketSend(comm->fd, &comm->ctag, sizeof(ucp_tag_t)));
  *send_comm = comm;
  free(my_addr);
  free(rkey_buf);

  return ncclSuccess;
}

#define REG_ALIGN (4096)
ncclResult_t ucx_regmr(void* comm, void* data, int size, int type, void** mhandle) {
  ucp_mem_map_params_t mmap_params;
  ucx_ctx *ctx = (ucx_ctx*)comm;
  ucx_mhandle *mh;
  
  uint64_t addr = (uint64_t)data;
  uint64_t reg_addr = addr & (~(REG_ALIGN-1));
  uint64_t reg_size = addr+size - reg_addr;
  reg_size = ((reg_size + REG_ALIGN-1) / REG_ALIGN ) * REG_ALIGN;

  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH; 
  mmap_params.address    = (void*)reg_addr;
  mmap_params.length     = reg_size;
  
  mh = (ucx_mhandle*)malloc(sizeof(ucx_mhandle));
  ucp_mem_map(ctx->ucp_ctx, &mmap_params, &mh->ucp_memh);
  if (ctx->gpuFlush.enabled) {
    size_t rkey_buf_size;
    void *rkey_buf;
    ucp_rkey_pack(ctx->ucp_ctx, mh->ucp_memh, &rkey_buf, &rkey_buf_size);
    UCXCHECK(ucp_ep_rkey_unpack(ctx->gpuFlush.flush_ep, rkey_buf, &mh->rkey));
  }
  
  *mhandle = mh;
  return ncclSuccess;
}

ncclResult_t ucx_deregmr(void* comm, void* mhandle) {
  ucx_ctx *ctx = (ucx_ctx*)comm;
  ucx_mhandle *mh = (ucx_mhandle*)mhandle;
  if (ctx->gpuFlush.enabled) {
      ucp_rkey_destroy(mh->rkey);
  }
  ucp_mem_unmap(ctx->ucp_ctx, mh->ucp_memh);
  free(mhandle);
  return ncclSuccess;
}

ncclResult_t ucx_accept(void *listen_comm, void **recv_comm) {
  ucx_recv_comm *r_comm = (ucx_recv_comm *)calloc(1, sizeof(ucx_recv_comm));
  ucx_listen_comm *l_comm = (ucx_listen_comm *)listen_comm;
  void *rkey_buf;
  size_t rkey_buf_size;

  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(l_comm->fd, (struct sockaddr_in *)&sockaddr, &socklen), "accept", r_comm->fd);

  r_comm->ctx = l_comm->ctx; r_comm->worker = l_comm->worker; r_comm->tag = l_comm->tag;
  NCCLCHECK(socketReceive(r_comm->fd, &rkey_buf_size, sizeof(size_t)));
  rkey_buf = malloc(rkey_buf_size);
  NCCLCHECK(socketReceive(r_comm->fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketReceive(r_comm->fd, &r_comm->rem_tail_addr, sizeof(uint64_t)));

  size_t peer_addr_len;
  ucp_address_t *peer_addr;
  ucp_ep_params_t ep_params;
  NCCLCHECK(socketReceive(r_comm->fd, &peer_addr_len, sizeof(size_t)));
  peer_addr = malloc(peer_addr_len);
  NCCLCHECK(socketReceive(r_comm->fd, peer_addr, peer_addr_len));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS; //|
  //                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = peer_addr;
  //  ep_params.err_mode        = err_handling_opt.ucp_err_mode;
  UCXCHECK(ucp_ep_create(r_comm->worker, &ep_params, &r_comm->ep));
  UCXCHECK(ucp_ep_rkey_unpack(r_comm->ep, rkey_buf, &r_comm->rkey));
  NCCLCHECK(socketReceive(r_comm->fd, &r_comm->ctag, sizeof(ucp_tag_t)));

  r_comm->gpuFlush.enabled = (ncclIbGdrSupport(l_comm->dev) == 0);  
  if (r_comm->gpuFlush.enabled) {
    ucp_address_t *my_addr;
    size_t local_addr_len;

    NCCLCHECK(ucx_worker_get_netaddress(r_comm->worker, &my_addr, &local_addr_len));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = my_addr;
    UCXCHECK(ucp_ep_create(r_comm->worker, &ep_params, &r_comm->gpuFlush.flush_ep));
  }
  free(peer_addr);
  free(rkey_buf);
  *recv_comm = r_comm;

  return ncclSuccess;
}

ncclResult_t ucx_send_check(ucx_send_comm *comm) {
  ucp_tag_message_h msg_tag;
  ucp_tag_recv_info_t info_tag;
  ucx_request *req;
  connect_msg *msg;
  ucp_ep_params_t ep_params;

  ucp_worker_progress(comm->worker);

  msg_tag = ucp_tag_probe_nb(comm->worker, comm->ctag, tag_mask, 1, &info_tag);
  if (msg_tag == NULL) {
    return ncclSuccess;
  }
  msg = malloc(info_tag.length);
  req = ucp_tag_msg_recv_nb(comm->worker, msg, info_tag.length, ucp_dt_make_contig(1), msg_tag, recv_handler);
  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("Unable to receive connect msg (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
  }
  else {
    while (req->completed == 0) {
      ucp_worker_progress(comm->worker);
    }
    req->completed = 0;
    ucp_request_release(req);
  }
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS; //|
  //                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = (ucp_address_t*)(msg + 1);
  //  ep_params.err_mode        = err_handling_opt.ucp_err_mode;
  UCXCHECK(ucp_ep_create(comm->worker, &ep_params, &comm->ep));
  comm->ready = 1;
  free(msg);

  return ncclSuccess;
}

ncclResult_t ucx_recv_check(ucx_recv_comm *comm) {
  if (comm->connect_req == NULL){
    ucp_address_t *my_addr;
    size_t local_addr_len;
    NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));
    size_t msg_len = sizeof(connect_msg) + local_addr_len;
    comm->msg = calloc(1, msg_len);
    comm->msg->addr_len = local_addr_len;
    memcpy(comm->msg + 1, my_addr, local_addr_len);
    comm->connect_req = ucp_tag_send_nb(comm->ep, comm->msg, msg_len, ucp_dt_make_contig(1), comm->ctag, send_handler);
    if (UCS_PTR_IS_ERR(comm->connect_req)) {
      NCCL_UCX_WARN("Unable to send connect message");
      return ncclSystemError;
    } else if (comm->connect_req == NULL){
      comm->ready = 1;
      free(comm->msg);
    }
    free(my_addr);
  }
  else{
    if (comm->connect_req->completed == 1) {
      comm->ready = 1;
      comm->connect_req->completed = 0;
      ucp_request_release(comm->connect_req);
      free(comm->msg);
    }
    else {
      ucp_worker_progress(comm->worker);
    }
  }
  return ncclSuccess;
}

ncclResult_t ucx_isend(void *send_comm, void *data, int size, void *mhandle, void **request) {
  ucx_request *req;
  ucx_send_comm *comm = (ucx_send_comm *)send_comm;

  if (comm->ready == 0) { ucx_send_check(comm); }
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }

  volatile uint32_t *head = &comm->fifo_head;
  volatile uint32_t *tail = &comm->fifo_tail;
  if (*head == *tail) { *request = NULL; return ncclSuccess; }
  req = ucp_tag_send_nb(comm->ep, data, size, ucp_dt_make_contig(1), comm->tag, send_handler);
  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("ucx_isend: unable to send message (%s)\n", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  }
  else if (req != NULL) {
    ucp_worker_progress(comm->worker);
    req->worker = comm->worker;
    req->size = size;
  }
  comm->fifo_head++;
  *request = req ? req : (size ? 1: 2);

  return ncclSuccess;
}

ncclResult_t ucx_irecv(void *recv_comm, void *data, int size, void *mhandle, void **request) {
  ucx_request *req;
  ucx_recv_comm *comm = (ucx_recv_comm *)recv_comm;

  if (comm->ready == 0) { ucx_recv_check(comm); }
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }
  req = ucp_tag_recv_nb(comm->worker, data, size, ucp_dt_make_contig(1), comm->tag, tag_mask, recv_handler);
  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("ucx_irecv: unable to receive message (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  }
  else if (req != NULL) {
    ucp_worker_progress(comm->worker);
    req->worker = comm->worker;
    req->size = size;
  }
  comm->tail++;
  ucp_put_nbi(comm->ep, &comm->tail, sizeof(uint32_t), comm->rem_tail_addr, comm->rkey);
  *request = req ? req : (size ? 1: 2);

  return ncclSuccess;
}

ncclResult_t ucx_test(void *request, int *done, int *size) {
  ucx_request *req = (ucx_request *)request;
  *done = 0;
  if ((uint64_t)request == 1ul || (uint64_t)request == 2ul) {
    *done = 1;
    if (size) *size = -1 + (uint64_t)request;
    return ncclSuccess;
  }
  if (req->completed == 1) {
    *done = 1;
    if(size) *size = req->size;
    req->completed = 0;
  }
  else {
    ucp_worker_progress(req->worker);
  }

  return ncclSuccess;
}


ncclResult_t ucx_flush(void* recv_comm, void* data, int size, void* mhandle) {
  ucx_recv_comm *comm = (ucx_recv_comm *)recv_comm;
  if (comm->gpuFlush.enabled == 0 || size == 0) return ncclSuccess;
  ucx_mhandle *mh = (ucx_mhandle*)mhandle;
  ucx_request *req;
  req = ucp_get_nb(comm->gpuFlush.flush_ep, &comm->gpuFlush.hostMem, 1, (uint64_t)data, mh->rkey, send_handler);
  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("ucx_flush: unable to read data (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  } else if (req != NULL){
    while(req->completed == 0){
      ucp_worker_progress(comm->worker);
    }
    req->completed = 0;
    ucp_request_release(req);
  }
  return ncclSuccess;
}

ncclResult_t ucx_close_send(void *send_comm) {
  if (send_comm){
    ucx_send_comm *comm = (ucx_send_comm*) send_comm;
    ucp_mem_unmap(comm->ctx, comm->fifo_memh);
    close(comm->fd);
    free(comm);
  //   void *close_req;
  //   ucs_status_t status;
  //   if (comm->ep){
  //     close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
  //     if (UCS_PTR_IS_PTR(close_req)){
  //       do{
  //         ucp_worker_progress(comm->worker);
  //         status = ucp_request_check_status(close_req);
  //       }while(status == UCS_INPROGRESS);
  //       ucp_request_free(close_req);
  //     } else if (close_req != NULL){
  //       NCCL_UCX_WARN("Failed to close UCX endpoint");
  //     }
  //   }
  //   ucp_worker_destroy(comm->worker);
  //   ucp_cleanup(comm->ctx);
  //   int done = 1;
  //   socketSend(comm->fd, &done, sizeof(int));
  }

  return ncclSuccess;
}

ncclResult_t ucx_close_recv(void *recv_comm) {
  if (recv_comm){
    ucx_recv_comm *comm = (ucx_recv_comm*)recv_comm;
    ucp_rkey_destroy(comm->rkey);
    close(comm->fd);
    free(recv_comm);
  //   ucp_worker_h ucp_worker;
  //   int peer_close_send;
  //   socketReceive(comm->fd, &peer_close_send, sizeof(int));
  //   ucp_worker = comm->worker;
  //   ucp_worker_destroy(ucp_worker);
  //   ucp_cleanup(comm->ctx);
  //   
  }
  
  return ncclSuccess;
}

ncclResult_t ucx_close_listen(void *listen_comm) {
  ucx_listen_comm *comm = (ucx_listen_comm *)listen_comm;
  if (comm) {
    close(comm->fd);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclNet_t NCCL_PLUGIN_SYMBOL = {
    "UCX",
    ucx_init,
    ucx_devices,
    ucx_pci_path,
    ucx_ptr_support,
    ucx_listen,
    ucx_connect,
    ucx_accept,
    ucx_regmr,
    ucx_deregmr,
    ucx_isend,
    ucx_irecv,
    ucx_flush,
    ucx_test,
    ucx_close_send,
    ucx_close_recv,
    ucx_close_listen
    };
