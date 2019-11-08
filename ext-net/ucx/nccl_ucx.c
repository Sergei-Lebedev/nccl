
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
#include "ibvwrap.h"
#include "nccl_ucx.h"


#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>

pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;

struct ucx_listen_handle{
  union socketAddress connectAddr;
};
typedef struct ucx_listen_handle ucx_listen_handle;

struct ucx_listen_comm{
  int fd;
  int dev;
};
typedef struct ucx_listen_comm ucx_listen_comm;

struct ucx_request {
  int completed;
  ucp_worker_h worker;
};
typedef struct ucx_request ucx_request;

struct nccl_ucx_worker{
  ucp_context_h ctx;
  ucp_worker_h worker;
  int count;
};
static struct nccl_ucx_worker workers[MAX_IB_DEVS];

struct ucx_send_comm {
  ucp_context_h ctx;
  ucp_worker_h worker;
  ucp_ep_h ep;
  int fd;
  int ready;
  ucp_tag_t tag;
};
typedef struct ucx_send_comm ucx_send_comm;

struct ucx_recv_comm {
  ucp_context_h ctx;
  ucp_worker_h worker;
  int fd;
  int ready;
  ucp_tag_t tag;
};
typedef struct ucx_recv_comm ucx_recv_comm;

static void request_init(void *request){
    struct ucx_request *req = (struct ucx_request *) request;
    req->completed = 0;
}

static void send_handler(void *request, ucs_status_t status){
    struct ucx_request *req = (struct ucx_request *) request;
    req->completed = 1;
    //   printf("[0x%x] send handler called with status %d (%s)\n", (unsigned int)pthread_self(), status, ucs_status_string(status));
}

static void failure_handler(void *arg, ucp_ep_h ep, ucs_status_t status){
    ucs_status_t *arg_status = (ucs_status_t *)arg;
    //    printf("[0x%x] failure handler called with status %d (%s)\n",(unsigned int)pthread_self(), status, ucs_status_string(status));
    *arg_status = status;
}

static void recv_handler(void *request, ucs_status_t status, ucp_tag_recv_info_t *info){
    struct ucx_request *req = (struct ucx_request *) request;
    req->completed = 1;
    //printf("[0x%x] receive handler called with status %d (%s), length %lu\n", (unsigned int)pthread_self(), status, ucs_status_string(status), info->length);
}

static union socketAddress nccl_ucx_if_addr;
static char if_name[MAX_IF_NAME_SIZE];

static ncclResult_t get_socket_addr(union socketAddress* addr) {
  memcpy(addr, &nccl_ucx_if_addr, sizeof(*addr));
  return ncclSuccess;
}

static ncclResult_t ucx_init_context(ucp_context_h *ctx, int dev){
  ucp_params_t ucp_params;
  ucp_config_t *config;
  
  char ucx_prefix[PATH_MAX]; //DEV_####
  char ucx_env_var[PATH_MAX]; //UCX_DEV_####_NET_DEVICES
  char ucx_env_val[PATH_MAX]; //e.g. mlx5_0:1
    
  snprintf(ucx_prefix, PATH_MAX, "DEV_%d", dev);
  snprintf(ucx_env_var, PATH_MAX, "UCX_%s_NET_DEVICES", ucx_prefix);
  snprintf(ucx_env_val, PATH_MAX, "%s:%d" , ncclIbDevs[dev].devName, ncclIbDevs[dev].port);
  setenv(ucx_env_var, ucx_env_val, 0);
  UCXCHECK(ucp_config_read(ucx_prefix, NULL, &config));
    
  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_REQUEST_SIZE |
                          UCP_PARAM_FIELD_REQUEST_INIT;
  ucp_params.features = UCP_FEATURE_TAG;
  ucp_params.request_size = sizeof(struct ucx_request);
  ucp_params.request_init = request_init;
  UCXCHECK(ucp_init(&ucp_params, config, ctx));
  ucp_config_release(config);
}

static ncclResult_t ucx_init_worker(ucp_context_h ctx, ucp_worker_h *worker){
  ucp_worker_params_t worker_params;
  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  UCXCHECK(ucp_worker_create(ctx, &worker_params, worker));
}

static ncclResult_t ucx_get_ctx_and_worker(int dev, ucp_context_h *ctx, ucp_worker_h *worker, ucp_tag_t *newtag){
  pthread_mutex_lock(&ncclIbLock);
  if (ncclNIbDevs < dev){
    NCCL_UCX_WARN("Device index is too large");
    return ncclSystemError;
  }
  if (workers[dev].count == 0){
    ucx_init_context(&workers[dev].ctx, dev);
    ucx_init_worker(workers[dev].ctx, &workers[dev].worker);
  }
  *ctx = workers[dev].ctx;
  *worker = workers[dev].worker;
  if (newtag != NULL){
    *newtag = tag + workers[dev].count;
  }
  workers[dev].count++;
  pthread_mutex_unlock(&ncclIbLock);
  return ncclSuccess;
}


ncclResult_t ucx_init(ncclDebugLogger_t logFunction) {
  ucx_log_function = logFunction;
  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      if (findInterfaces(if_name, &nccl_ucx_if_addr, MAX_IF_NAME_SIZE, 1) != 1) {
        NCCL_UCX_WARN("NET/UCX : No IP interface found.");
        return ncclInternalError;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      char* userIbEnv = getenv("NCCL_IB_HCA");
      struct netIf userIfs[MAX_IB_DEVS];
      int searchNot = userIbEnv && userIbEnv[0] == '^';
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) return ncclInternalError;

      for (int d=0; d<nIbDevs; d++) {
        struct ibv_context * context;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          NCCL_UCX_WARN("NET/UCX : Unable to open device %s", devices[d]->name);
          continue;
        }
        int found = 0;
        struct ibv_device_attr devAttr;
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          NCCL_UCX_WARN("NET/UCX : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
          continue;
        }
        for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
          struct ibv_port_attr portAttr;
          if (ncclSuccess != wrap_ibv_query_port(context, port, &portAttr)) {
            WARN("NET/UCX : Unable to query port %d", port);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE) continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

          // check against user specified HCAs/ports
          if (! (matchIfList(devices[d]->name, port, userIfs, nUserIfs) ^ searchNot)) {
            continue;
          }
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].context = context;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          //ucx_init_context(&workers[ncclNIbDevs].ctx, ncclNIbDevs);
          //ucx_init_worker(workers[ncclNIbDevs].ctx, &workers[ncclNIbDevs].worker);
          workers[ncclNIbDevs].count=0;
          ncclNIbDevs++;
          found++;
          //pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, context);
        }
        if (found == 0 && ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
      }
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { return ncclInternalError; };
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/UCX : No device found.");
    } else {
      char line[1024];
      line[0] = '\0';
      for (int d=0; d<ncclNIbDevs; d++) {
        snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
                 ncclIbDevs[d].port, ncclIbDevs[d].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
      }
      line[1023] = '\0';
      char addrline[1024];
      INFO(NCCL_INIT|NCCL_NET, "NET/UCX : Using%s ; OOB %s", line, if_name);
    }
    pthread_mutex_unlock(&ncclIbLock);
  }
  return ncclSuccess;
}

ncclResult_t ucx_listen(int dev, void* handle, void** listen_comm) {
  ucx_listen_handle *my_handle;
  ucx_listen_comm *comm;

  //allocate listen comm which contains ucp_worker and socket address to exchange
  comm = malloc(sizeof(ucx_listen_comm));
  memset(comm, 0, sizeof(ucx_listen_comm));
  
  static_assert(sizeof(ucx_listen_handle) < NCCL_NET_HANDLE_MAXSIZE, "ucx listen handle size too large");
  my_handle = (ucx_listen_handle*)handle;
  comm->dev = dev;
  //TODO: add error checking
  get_socket_addr(&(my_handle->connectAddr));
  createListenSocket(&comm->fd, &my_handle->connectAddr);
  *listen_comm = comm;

  return ncclSuccess;
}

ncclResult_t ucx_connect(int dev, void* handle, void** send_comm) {
  ucp_worker_params_t worker_params;

  ucx_listen_handle *recv_handle = (ucx_listen_handle*)handle;
  ucx_send_comm     *comm        = (ucx_send_comm*)malloc(sizeof(ucx_send_comm));

  //TODO: add error checking
  connectAddress(&comm->fd, &recv_handle->connectAddr);
  ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker, NULL);
  //ucx_init_context(&comm->ctx, dev);
  //ucx_init_worker(comm->ctx, &comm->worker);
  
  comm->ready = 0;
  comm->ep    = NULL;
  *send_comm = comm;
  return ncclSuccess;
}

ncclResult_t ucx_accept(void* listen_comm, void** recv_comm) {
  ucx_recv_comm *r_comm = (ucx_recv_comm*)malloc(sizeof(ucx_recv_comm));
  ucx_listen_comm *l_comm = (ucx_listen_comm*)listen_comm;

  r_comm->ready = 0;
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(l_comm->fd, (struct sockaddr_in*)&sockaddr, &socklen), "accept", r_comm->fd);

  // get worker address and send it to peer
  ucp_address_t *my_addr;
  size_t local_addr_len;

  ucx_get_ctx_and_worker(l_comm->dev, &r_comm->ctx, &r_comm->worker, &r_comm->tag);
  //ucx_init_context(&r_comm->ctx, l_comm->dev);
  //ucx_init_worker(r_comm->ctx, &r_comm->worker); r_comm->tag = tag;
  UCXCHECK(ucp_worker_get_address(r_comm->worker, &my_addr, &local_addr_len));
  NCCLCHECK(socketSend(r_comm->fd, &local_addr_len, sizeof(size_t)));
  NCCLCHECK(socketSend(r_comm->fd, my_addr, local_addr_len));
  NCCLCHECK(socketSend(r_comm->fd, &r_comm->tag, sizeof(ucp_tag_t)));
  ucp_worker_release_address(r_comm->worker, my_addr);
  NCCL_UCX_INFO(NCCL_NET, "Worker address length: %zu", local_addr_len);

  *recv_comm = r_comm;

  return ncclSuccess;
}

ncclResult_t ucx_send_check(ucx_send_comm *comm){
  int bytes = 0;
  size_t peer_addr_len;
  ucp_address_t *peer_addr;
  ucp_ep_h ep;
  ucp_ep_params_t ep_params;

  NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, comm->fd, &peer_addr_len, sizeof(size_t), &bytes));
  if (bytes == 0){
    return ncclSuccess;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, comm->fd, &peer_addr_len, sizeof(size_t), &bytes));
  peer_addr = malloc(peer_addr_len);
  bytes = 0;
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, comm->fd, peer_addr, peer_addr_len, &bytes));
  
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;//|
  //                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = peer_addr;
  //  ep_params.err_mode        = err_handling_opt.ucp_err_mode;
  //  NCCL_UCX_INFO(NCCL_NET, "Worker: %p", ucp_worker);
  UCXCHECK(ucp_ep_create(comm->worker, &ep_params, &ep));
  bytes = 0;
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, comm->fd, &comm->tag, sizeof(ucp_tag_t), &bytes));
  comm->ep    = ep;
  comm->ready = 1;
  free(peer_addr);
  NCCLCHECK(socketSend(comm->fd, &comm->ready, sizeof(int)));
  return ncclSuccess;
}

ncclResult_t ucx_recv_check(ucx_recv_comm *comm){
  int bytes = 0;
  NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, comm->fd, &comm->ready, sizeof(int), &bytes));
  if (bytes == 0){
    return ncclSuccess;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, comm->fd, &comm->ready, sizeof(int), &bytes));
  return ncclSuccess;
}

ncclResult_t ucx_isend(void* send_comm, void* data, int size, void* mhandle, void** request) {
  ucx_request *req;
  ucx_send_comm *comm = (ucx_send_comm*) send_comm;
  if (comm->ready == 0){ ucx_send_check(comm);}
  if (comm->ready == 0) { *request = NULL; return ncclSuccess;}
  
  req = ucp_tag_send_nb(comm->ep, data, size, ucp_dt_make_contig(1), comm->tag, send_handler);
  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("ucx_isend: unable to send message (%s)\n", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  }
  else if (req!= NULL){
    ucp_worker_progress(comm->worker);
    req->worker = comm->worker;
  }  
  *request = req ? req : 1;
  return ncclSuccess;
}

ncclResult_t ucx_irecv(void* recv_comm, void* data, int size, void* mhandle, void** request) {
  ucx_request *req;
  ucp_worker_h ucp_worker;
  ucx_recv_comm *comm = (ucx_recv_comm*) recv_comm;
  //if (comm->ready == 0) { ucx_recv_check(comm);}
  //if (comm->ready == 0) { *request = NULL; return ncclSuccess;}
  
  ucp_worker = comm->worker;
  req = ucp_tag_recv_nb(ucp_worker, data, size, ucp_dt_make_contig(1), comm->tag, tag_mask, recv_handler);
  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("ucx_irecv: unable to receive message (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  }else if (req != NULL){
    ucp_worker_progress(comm->worker);
    req->worker = ucp_worker;
  }
  *request = req ? req : 1;
  return ncclSuccess;
}

ncclResult_t ucx_test(void* request, int* done, int* size) {
  ucx_request *req = (ucx_request*)request;
  *done = 0;
  // we don't set size cause we don't use it later in ucx_flush
  //  if(size) *size = 0;
  if (request == 1){
    *done = 1;
    return ncclSuccess;
  }
  if (req->completed == 1){
    *done = 1;
    req->completed = 0;
    ucp_request_release(req);
  } else {
    ucp_worker_progress(req->worker);
  }
  return ncclSuccess;
}

ncclResult_t ucx_close_send(void* send_comm) {
  // if (send_comm){
  //   ucx_send_comm *comm;
  //   void *close_req;
  //   ucs_status_t status;
  //   comm = (ucx_send_comm*) send_comm;
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
  //   close(comm->fd);
  //   free(comm);
  // }
  return ncclSuccess;
}

ncclResult_t ucx_close_recv(void* recv_comm) {
  // if (recv_comm){
  //   ucp_worker_h ucp_worker;
  //   ucx_recv_comm *comm = (ucx_recv_comm*)recv_comm;
  //   int peer_close_send;
  //   socketReceive(comm->fd, &peer_close_send, sizeof(int));
  //   ucp_worker = comm->worker;
  //   ucp_worker_destroy(ucp_worker);
  //   ucp_cleanup(comm->ctx);
  //   close(comm->fd);
  //   free(recv_comm);
  // }
  return ncclSuccess;
}

ncclResult_t ucx_close_listen(void* listen_comm) {
  ucx_listen_comm *comm = (ucx_listen_comm*)listen_comm;
  if (comm){
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
