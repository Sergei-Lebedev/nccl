
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
#include "ib_utils.h"


#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>

pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;

struct ucx_listen_handle{
  union socketAddress peer_addr;
};
typedef struct ucx_listen_handle ucx_listen_handle;

struct ucx_listen_comm{
  ucp_worker_h worker;
  ucp_ep_h ep;
  ucp_listener_h listener;
  ucp_context_h ctx;
};
typedef struct ucx_listen_comm ucx_listen_comm;

struct ucx_request {
  int completed;
  ucp_worker_h worker;
};
typedef struct ucx_request ucx_request;

struct ucx_comm {
  ucp_context_h ctx;
  ucp_worker_h worker;
  ucp_ep_h ep;
};
typedef struct ucx_comm ucx_comm;

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

static void recv_handler(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info){
    struct ucx_request *req = (struct ucx_request *) request;
    req->completed = 1;
    //printf("[0x%x] receive handler called with status %d (%s), length %lu\n", (unsigned int)pthread_self(), status, ucs_status_string(status), info->length);
}

ncclResult_t ucx_init(ncclDebugLogger_t logFunction) {
  ucx_log_function = logFunction;
  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
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
          
          char ib_if_name[MAX_IF_NAME_SIZE];
          if (!dev2if(devices[d]->name, port, ib_if_name)) continue;

          struct sockaddr_storage addr;
          if (!get_ipoib_ip(ib_if_name, &addr)) continue;

          // check against user specified HCAs/ports
          if (! (matchIfList(devices[d]->name, port, userIfs, nUserIfs) ^ searchNot)) {
            continue;
          }
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].context = context;
          ncclIbDevs[ncclNIbDevs].addr = addr;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          ncclNIbDevs++;
          found++;
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
    }
    pthread_mutex_unlock(&ncclIbLock);
  }
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

static void conn_hand_cb(ucp_conn_request_h conn_request, void *arg){
  ucp_ep_h ep;
  ucp_ep_params_t ep_params;
  ucx_listen_comm *l_comm = (ucx_listen_comm *)arg;

  ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
  ep_params.conn_request = conn_request;
  UCXCHECK_VOID(ucp_ep_create(l_comm->worker, &ep_params, &(l_comm->ep)));
}

ncclResult_t ucx_listen(int dev, void* handle, void** listen_comm) {
  ucx_listen_handle *my_handle;
  ucx_listen_comm *comm;
  
  comm = malloc(sizeof(ucx_listen_comm));
  memset(comm, 0, sizeof(ucx_listen_comm));
  
  static_assert(sizeof(ucx_listen_handle) < NCCL_NET_HANDLE_MAXSIZE, "ucx listen handle size too large");
  my_handle = (ucx_listen_handle*)handle;

  ucx_init_context(&comm->ctx, dev);
  ucp_worker_params_t worker_params;
  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  UCXCHECK(ucp_worker_create(comm->ctx, &worker_params, &comm->worker));

  ucp_listener_params_t listener_params;
  listener_params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                               UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  int is_ipv4 = (ncclIbDevs[dev].addr.ss_family == AF_INET);
  struct sockaddr *listener_addr = (struct sockaddr*)(&ncclIbDevs[dev].addr);
  int addrlen = is_ipv4 ? sizeof(struct sockaddr_in): sizeof(struct sockaddr_in6);
  memcpy(&(my_handle->peer_addr), listener_addr, addrlen);
  if (is_ipv4){
    my_handle->peer_addr.sin.sin_port = 12131;
  }
  else {
    my_handle->peer_addr.sin6.sin6_port = 12131;
  }
  listener_params.sockaddr.addr = &(my_handle->peer_addr.sa);
  listener_params.sockaddr.addrlen = addrlen;
  listener_params.conn_handler.cb = conn_hand_cb;
  listener_params.conn_handler.arg = comm;
  UCXCHECK(ucp_listener_create(comm->worker, &listener_params, &comm->listener));
 
  char ipstr[INET6_ADDRSTRLEN];
  unsigned short int port;
  port = is_ipv4 ? my_handle->peer_addr.sin.sin_port:
                   my_handle->peer_addr.sin6.sin6_port;
  inet_ntop(is_ipv4 ? AF_INET: AF_INET6, is_ipv4 ? (void*)&((struct sockaddr_in*)listener_addr)->sin_addr :
                                                   (void*)&((struct sockaddr_in6*)listener_addr)->sin6_addr, ipstr, INET6_ADDRSTRLEN);
  NCCL_UCX_INFO(NCCL_NET, "UCX listener address: %s:%hu", ipstr, port);
  comm->ep = NULL;
  *listen_comm = comm;
  return ncclSuccess;
}

ncclResult_t ucx_connect(int dev, void* handle, void** send_comm) {
  ucp_worker_params_t worker_params;

  ucx_listen_handle *recv_handle = (ucx_listen_handle*)handle;
  ucx_comm     *comm        = (ucx_comm*)malloc(sizeof(ucx_comm));

  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  ucx_init_context(&comm->ctx, dev);
  UCXCHECK(ucp_worker_create(comm->ctx, &worker_params, &comm->worker));   

  ucp_ep_params_t ep_params;
  ucp_ep_h ep;

  int is_ipv4 = (recv_handle->peer_addr.sa.sa_family == AF_INET);
  int addrlen = is_ipv4 ? sizeof(struct sockaddr_in): sizeof(struct sockaddr_in6);

  ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS            |
                         UCP_EP_PARAM_FIELD_SOCK_ADDR        |
                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  
  ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
  ep_params.flags    = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
  ep_params.sockaddr.addr    = &(recv_handle->peer_addr.sa);
  ep_params.sockaddr.addrlen = addrlen;

  UCXCHECK(ucp_ep_create(comm->worker, &ep_params, &ep));

  comm->ep    = ep;
  *send_comm = comm;

  return ncclSuccess;
}

ncclResult_t ucx_accept(void* listen_comm, void** recv_comm) {
  //  NCCL_UCX_INFO(NCCL_NET, "ucx_accept");
  ucx_comm *r_comm = (ucx_comm*)malloc(sizeof(ucx_comm));
  ucx_listen_comm *l_comm = (ucx_listen_comm*)listen_comm;

  r_comm->worker = l_comm->worker;
  r_comm->ctx = l_comm->ctx;

  while(l_comm->ep == NULL){
    ucp_worker_progress(l_comm->worker);
  }
  r_comm->ep = l_comm->ep;
  *recv_comm = r_comm;

  return ncclSuccess;
}

ncclResult_t ucx_isend(void* send_comm, void* data, int size, void* mhandle, void** request) {
  ucx_request *req;
  ucx_comm *comm = (ucx_comm*) send_comm;

  req = ucp_tag_send_nb(comm->ep, data, size, ucp_dt_make_contig(1), tag, send_handler);
  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("ucx_isend: unable to send message\n");
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
  ucx_comm *comm = (ucx_comm*) recv_comm;
  
  ucp_worker = comm->worker;
  req = ucp_tag_recv_nb(ucp_worker, data, size, ucp_dt_make_contig(1), tag, tag_mask, recv_handler);
  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("ucx_irecv: unable to receive message (%s)",
            ucs_status_string(UCS_PTR_STATUS(req)));
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
  if (send_comm){
    ucx_comm *comm;
    void *close_req;
    ucs_status_t status;
    comm = (ucx_comm*) send_comm;
    if (comm->ep){
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      if (UCS_PTR_IS_PTR(close_req)){
        do{
          ucp_worker_progress(comm->worker);
          status = ucp_request_check_status(close_req);
        }while(status == UCS_INPROGRESS);
        ucp_request_free(close_req);
      } else if (close_req != NULL){
        NCCL_UCX_WARN("Failed to close UCX endpoint");
      }
    }
    ucp_worker_destroy(comm->worker);
    ucp_cleanup(comm->ctx);
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ucx_close_recv(void* recv_comm) {
  if (recv_comm){
    ucp_worker_h ucp_worker;
    ucx_comm *comm = (ucx_comm*)recv_comm;
    ucp_worker = comm->worker;
    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(comm->ctx);
    free(recv_comm);
  }
  return ncclSuccess;
}

ncclResult_t ucx_close_listen(void* listen_comm) {
  ucx_listen_comm *comm = (ucx_listen_comm*)listen_comm;
  if (comm){
    ucp_listener_destroy(comm->listener);
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
