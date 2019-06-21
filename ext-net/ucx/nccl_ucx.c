/*************************************************************************
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#include "nccl.h"
#include "nccl_net.h"
#include "core.h"
#include "socket.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <ucp/api/ucp.h>
#include <nccl_ucx_log.h>

#define __hidden __attribute__ ((visibility("hidden")))

#define UCP_WORKER_NONE  0xABBAABBA

#define UCXCHECK(cmd) do {                               \
  int e = cmd;                                           \
  if( UCS_OK != e ) {                                    \
    NCCL_UCX_WARN("Failed: UCX error %s:%d '%d' %s\n",   \
        __FILE__,__LINE__, e, ucs_status_string(e));     \
    return ncclInternalError;                            \
  }                                                      \
} while(0)


ncclDebugLogger_t ucx_log_function = NULL;
static const ucp_tag_t tag  = 0xABADBABE;
static const ucp_tag_t tag_mask = 0;//0xFFFFFFFFFFFFFFFF; //recieve any message

ucp_context_h ucp_context;
ucp_worker_h ucp_worker;

struct ucx_listen_handle{
  union socketAddress connectAddr;
  size_t local_addr_len;
};
typedef struct ucx_listen_handle ucx_listen_handle;

struct ucx_listen_comm{
  int fd;
  ucx_listen_handle handle;
  ucp_worker_h worker;
};
typedef struct ucx_listen_comm ucx_listen_comm;

struct ucx_request {
  int size;
  int completed;
  int release;
  int type;
  ucp_worker_h worker;
};
typedef struct ucx_request ucx_request;

struct ucx_send_comm {
  ucp_worker_h worker;
  ucp_ep_h ep;
  int fd;
};
typedef struct ucx_send_comm ucx_send_comm;

struct ucx_recv_comm {
  ucp_worker_h worker;
  int fd;
};
typedef struct ucx_recv_comm ucx_recv_comm;

static void request_init(void *request){
    struct ucx_request *req = (struct ucx_request *) request;
    req->completed = 0;
}

static void send_handler(void *request, ucs_status_t status){
    struct ucx_request *req = (struct ucx_request *) request;
    req->completed = 1;
    req->release =  1;
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
    req->release = 1;
    //printf("[0x%x] receive handler called with status %d (%s), length %lu\n", (unsigned int)pthread_self(), status, ucs_status_string(status), info->length);
}


static union socketAddress nccl_ucx_if_addr;
static char if_name[MAX_IF_NAME_SIZE];

static ncclResult_t get_socket_addr(union socketAddress* addr) {
  memcpy(addr, &nccl_ucx_if_addr, sizeof(*addr));
  return ncclSuccess;
}

static pthread_mutex_t worker_mutex;

__hidden ncclResult_t ucx_init(ncclDebugLogger_t logFunction) {
  int status;
  ucp_params_t ucp_params;
  ucp_config_t *config;

  ucx_log_function = logFunction;
  UCXCHECK(ucp_config_read(NULL, NULL, &config));
  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_REQUEST_SIZE |
                          UCP_PARAM_FIELD_REQUEST_INIT;
  ucp_params.features = UCP_FEATURE_TAG;
  ucp_params.request_size = sizeof(struct ucx_request);
  ucp_params.request_init = request_init;
  UCXCHECK(ucp_init(&ucp_params, config, &ucp_context));
  ucp_worker = (ucp_worker_h)(uintptr_t)UCP_WORKER_NONE;
  ucp_config_release(config);

  if (findInterfaces(if_name, &nccl_ucx_if_addr, MAX_IF_NAME_SIZE, 1) != 1) {
        NCCL_UCX_WARN("NET/UCX : No IP interface found.");
        return ncclInternalError;
   }
  NCCL_UCX_INFO(NCCL_NET, "Using OOB %s", if_name);

  //pthread_mutex_init(&worker_mutex, NULL);
  return ncclSuccess;
}

__hidden ncclResult_t ucx_devices(int* ndev) {
  *ndev = 1; return ncclSuccess;
}

__hidden ncclResult_t ucx_pci_path(int dev, char** path) {
  char devicepath[PATH_MAX];
  snprintf(devicepath, PATH_MAX, "/sys/class/infiniband/%s/device", "mlx5_0");
  *path = realpath(devicepath, NULL);
  if (*path == NULL) {
    NCCL_UCX_WARN("Could not find real path of %s", devicepath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

__hidden ncclResult_t ucx_ptr_support(int dev, int* supported_types) {
  *supported_types = (NCCL_PTR_HOST | NCCL_PTR_CUDA);
  //  *supported_types = NCCL_PTR_HOST;
  return ncclSuccess;
}

__hidden ncclResult_t ucx_listen(int dev, void* handle, void** listen_comm) {
  ucp_worker_params_t worker_params;
  ucx_listen_handle *my_handle;
  ucx_listen_comm *comm;
  
  //  pthread_mutex_lock(&worker_mutex);
  //  NCCL_UCX_INFO(NCCL_NET, "ucx_listen");

  //allocate listen comm which contains ucp_worker and socket address to exchange
  comm = malloc(sizeof(ucx_listen_comm));
  memset(comm, 0, sizeof(ucx_listen_comm));
  
  static_assert(sizeof(ucx_listen_handle) < NCCL_NET_HANDLE_MAXSIZE, "ucx listen handle size too large");
  my_handle = (ucx_listen_handle*)handle;

  //create ucp_worker
  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  UCXCHECK(ucp_worker_create(ucp_context, &worker_params, &ucp_worker));

  //TODO: add error checking
  get_socket_addr(&(my_handle->connectAddr));
  createListenSocket(&comm->fd, &my_handle->connectAddr);

  comm->worker = ucp_worker;
  *listen_comm = comm;

  
  return ncclSuccess;
}

static void wait(ucp_worker_h ucp_worker, struct ucx_request *request){
    while (request->completed == 0) {
        ucp_worker_progress(ucp_worker);
    }
}

__hidden ncclResult_t ucx_connect(int dev, void* handle, void** send_comm) {
  ucp_ep_h ep;
  ucp_ep_params_t ep_params;
  size_t peer_addr_len;
  ucp_address_t *peer_addr;

  ucx_listen_handle *recv_handle = (ucx_listen_handle*)handle;
  ucx_send_comm     *comm        = (ucx_send_comm*)malloc(sizeof(ucx_send_comm));

  //TODO: add error checking
  connectAddress(&comm->fd, &recv_handle->connectAddr);

  //recieve worker address from peer
  socketReceive(comm->fd, &peer_addr_len, sizeof(size_t));
  peer_addr = malloc(peer_addr_len);
  socketReceive(comm->fd, peer_addr, peer_addr_len);
  
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;//|
    //                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = peer_addr;
  //  ep_params.err_mode        = err_handling_opt.ucp_err_mode;
  //  NCCL_UCX_INFO(NCCL_NET, "Worker: %p", ucp_worker);

  //check if ucp worker exists 
  if (ucp_worker == (ucp_worker_h)(uintptr_t)UCP_WORKER_NONE) {
    ucp_worker_params_t worker_params;
    
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    UCXCHECK(ucp_worker_create(ucp_context, &worker_params, &ucp_worker));   
  }
  UCXCHECK(ucp_ep_create(ucp_worker, &ep_params, &ep));


  comm->worker = ucp_worker;
  comm->ep     = ep;
  *send_comm   = comm;
  ucp_worker = (ucp_worker_h)(uintptr_t)UCP_WORKER_NONE;
  free(peer_addr);
  //pthread_mutex_unlock(&worker_mutex);
  return ncclSuccess;
}

__hidden ncclResult_t ucx_accept(void* listen_comm, void** recv_comm) {
  //  NCCL_UCX_INFO(NCCL_NET, "ucx_accept");
  ucx_recv_comm *r_comm = (ucx_recv_comm*)malloc(sizeof(ucx_recv_comm));
  ucx_listen_comm *l_comm = (ucx_listen_comm*)listen_comm;
  
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(l_comm->fd, (struct sockaddr_in*)&sockaddr, &socklen), "accept", r_comm->fd);

  // get worker address and send it to peer
  ucp_address_t *my_addr;
  size_t local_addr_len;
  r_comm->worker = l_comm->worker;
  UCXCHECK(ucp_worker_get_address(r_comm->worker, &my_addr, &local_addr_len));
  NCCL_UCX_INFO(NCCL_NET, "WA len: %zu", local_addr_len);
  socketSend(r_comm->fd, &local_addr_len, sizeof(size_t));
  socketSend(r_comm->fd, my_addr, local_addr_len);
  ucp_worker_release_address(ucp_worker, my_addr);
  *recv_comm = r_comm;


  return ncclSuccess;
}

__hidden ncclResult_t ucx_regmr(void* comm, void* data, int size, int type, void** mhandle){
  ucp_mem_map_params_t mmap_params;
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
    UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    //                           UCP_MEM_MAP_PARAM_FIELD_FLAGS;
  mmap_params.address    = (void*)(uint64_t)data;
  mmap_params.length     = size;
  mmap_params.flags      = UCP_MEM_MAP_FIXED;
  //  ucp_mem_map(ucp_context, &mmap_params, mhandle);
  return ncclSuccess;
}

__hidden ncclResult_t ucx_deregmr(void* comm, void* mhandle){
  //  ucp_mem_unmap(ucp_context, mhandle);
  return ncclSuccess;
}

__hidden ncclResult_t ucx_isend(void* send_comm, void* data, int size, void* mhandle, void** request) {
  ucx_request *req;
  ucx_send_comm *comm;

  comm = (ucx_send_comm*) send_comm;

  req = ucp_tag_send_nb(comm->ep, data, size, ucp_dt_make_contig(1), tag, send_handler);
  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("ucx_isend: unable to send UCX address message\n");
    return ncclSystemError;
  }
  else if (UCS_PTR_STATUS(req) == UCS_OK){
    req = (ucx_request *)malloc(sizeof(ucx_request));
    req->completed = 1;
    req->release = 0;
  }
  else {
      req->release = 1;
      req->worker = comm->worker;
  }
  
  req->type = 0;
  req->size = size;
  *request = req;
  return ncclSuccess;
}

__hidden ncclResult_t ucx_irecv(void* recv_comm, void* data, int size, void* mhandle, void** request) {
  ucx_request *req;
  ucp_worker_h ucp_worker;
  ucx_recv_comm *comm = (ucx_recv_comm*) recv_comm;

  ucp_worker = comm->worker;
  req = ucp_tag_recv_nb(ucp_worker, data, size, ucp_dt_make_contig(1), tag, tag_mask, recv_handler);

  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("ucx_irecv: unable to receive UCX address message (%s)\n",
            ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  }else if (UCS_PTR_STATUS(req) == UCS_OK){
    req = (ucx_request *)malloc(sizeof(ucx_request));
    req->completed = 1;
    req->release = 0;
  }
  else {
    req->release = 1;
    req->worker = ucp_worker;
  }
  req->type = 1;
  req->size = size;
  *request = req;
  return ncclSuccess;
}

__hidden ncclResult_t ucx_flush(void* recv_comm, void* data, int size, void* mhandle) {
  ucp_worker_h ucp_worker;
  ucx_recv_comm *comm = (ucx_recv_comm*)recv_comm;
  ucp_worker = comm->worker;

  ucx_request *req;
  req = ucp_worker_flush_nb(ucp_worker, 0, send_handler);
  if (UCS_PTR_IS_ERR(req)) {
    fprintf(stderr, "Error executing ucp_worker_flush operation\n");
  } else if (UCS_PTR_STATUS(req) != UCS_OK) {
    wait(ucp_worker, req);
    req->completed = 0; req->release = 0;
    ucp_request_release(req);
  }
  return ncclSuccess;
}

__hidden ncclResult_t ucx_test(void* request, int* done, int* size) {
  ucx_request *req = (ucx_request*)request;
  *done = 0;

  if(size) *size = 0;
  if (req->completed == 1){
    *done = 1;
    if (size) *size = req->size;
    if (req->release){
      req->size = 0;
      req->completed = 0;
      req->release = 0;
      ucp_request_release(req);
    }
    return ncclSuccess;
  } else {
    ucp_worker_progress(req->worker);
  }
  return ncclSuccess;
}

__hidden ncclResult_t ucx_close_send(void* send_comm) {
  ucx_send_comm *comm;
  void *close_req;
  ucs_status_t status;
  
  comm = (ucx_send_comm*) send_comm;
  close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
  if (UCS_PTR_IS_PTR(close_req)){
    do{
      ucp_worker_progress(comm->worker);
      status = ucp_request_check_status(close_req);
    }while(status == UCS_INPROGRESS);
    ucp_request_free(close_req);
  } else if (UCS_PTR_STATUS(close_req) != UCS_OK){
    NCCL_UCX_WARN("Failed to close UCX endpoint");
  }
  
  ucp_worker_destroy(comm->worker);
  comm->worker = NULL;
  return ncclSuccess;
}

__hidden ncclResult_t ucx_close_recv(void* recv_comm) {
  ucp_worker_h ucp_worker;
  ucp_worker = ((ucx_recv_comm*)recv_comm)->worker;
  if (ucp_worker){
    ucp_worker_destroy(ucp_worker);
  }
  ucp_cleanup(ucp_context);
  return ncclSuccess;
}

__hidden ncclResult_t ucx_close_listen(void* listen_comm) {
  ucx_listen_comm *comm = (ucx_listen_comm*)listen_comm;

  if (comm){
    close(comm->fd);
    free(listen_comm);
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
