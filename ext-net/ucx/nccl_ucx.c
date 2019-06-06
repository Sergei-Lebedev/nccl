/*************************************************************************
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <nccl.h>
#include <nccl_net.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <ucp/api/ucp.h>
#include <nccl_ucx_log.h>

#define __hidden __attribute__ ((visibility("hidden")))

#define UCP_WORKER_NONE  0xABBAABBA
#define TAG_SEND    1
#define STREAM_SEND 2
#define SEND_TYPE        TAG_SEND

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
static const ucp_tag_t tag_mask = 0; //recieve any message

ucp_context_h ucp_context;
ucp_worker_h ucp_worker;

struct ucx_listen_handle{
  size_t local_addr_len;
  long hostid;
};
typedef struct ucx_listen_handle ucx_listen_handle;

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
};
typedef struct ucx_send_comm ucx_send_comm;

struct ucx_recv_comm {
  ucp_worker_h worker;
#if SEND_TYPE == STREAM_SEND
  ucp_ep_h ep;
#endif
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

static void recv_handler_stream(void *request, ucs_status_t status, size_t length){
    struct ucx_request *req = (struct ucx_request *) request;
    req->completed = 1;
    req->release = 1;
    //    printf("[0x%x] receive handler called with status %d (%s), length %lu\n", (unsigned int)pthread_self(), status, ucs_status_string(status), length);
}

static pthread_mutex_t worker_mutex;

__hidden ncclResult_t ucx_init(ncclDebugLogger_t logFunction) {
  int status;
  ucp_params_t ucp_params;
  ucp_config_t *config;

  int debug = 0;
  while(debug)
    {
      debug =1;
    }

  ucx_log_function = logFunction;
  UCXCHECK(ucp_config_read(NULL, NULL, &config));
  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_REQUEST_SIZE |
                          UCP_PARAM_FIELD_REQUEST_INIT;
  ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_STREAM;
  ucp_params.request_size = sizeof(struct ucx_request);
  ucp_params.request_init = request_init;
  UCXCHECK(ucp_init(&ucp_params, config, &ucp_context));
  ucp_worker = NULL;
  ucp_config_release(config);
  pthread_mutex_init(&worker_mutex, NULL);
  //  fprintf(stderr, "ucx_init\n");
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
  return ncclSuccess;
}

__hidden ncclResult_t ucx_listen(int dev, void* handle, void** listen_comm) {
  ucp_worker_params_t worker_params;
  ucx_listen_handle *my_handle;
  ucp_address_t *my_addr;

  pthread_mutex_lock(&worker_mutex);
  //  NCCL_UCX_INFO(NCCL_NET, "ucx_listen");
  my_handle = (ucx_listen_handle*)handle;
  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  UCXCHECK(ucp_worker_create(ucp_context, &worker_params, &ucp_worker));
  UCXCHECK(ucp_worker_get_address(ucp_worker, &(my_addr), &(my_handle->local_addr_len)));
  memcpy(my_handle+1, my_addr, my_handle->local_addr_len);
  my_handle->hostid = gethostid();
  //  NCCL_UCX_INFO(NCCL_NET,"[0x%x][hostid: %lu] local address length: %lu\n", (unsigned int)pthread_self(), my_handle->hostid, my_handle->local_addr_len);
  *listen_comm = (void*)ucp_worker;
  ucp_worker_release_address(ucp_worker, my_addr);
  
  return ncclSuccess;
}

static void wait(ucp_worker_h ucp_worker, struct ucx_request *request){
    while (request->completed == 0) {
        ucp_worker_progress(ucp_worker);
    }
}

__hidden ncclResult_t ucx_connect(int dev, void* handle, void** send_comm) {
  //ucp_ep_h *ep;
  ucp_ep_h ep;
  ucp_ep_params_t ep_params;
  ucx_listen_handle *recv_handle = (ucx_listen_handle*)handle;
  long hostid = gethostid();
  //  fprintf(stderr, "My host id: %lu, recieved host id: %lu, worker %p\n", hostid, recv_handle->hostid, ucp_worker);
  //  NCCL_UCX_INFO(NCCL_NET,"My host id: %lu, recieved host id: %lu, worker %p\n", hostid, recv_handle->hostid, ucp_worker);
  size_t peer_addr_len;
  ucp_address_t *peer_addr;

  //  ep = (ucp_ep_h*)malloc(sizeof(ucp_ep_h));
  
  peer_addr_len = recv_handle->local_addr_len;
  peer_addr = malloc(recv_handle->local_addr_len);
  memcpy(peer_addr, recv_handle+1, recv_handle->local_addr_len);
  
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;//|
    //                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = peer_addr;
  //  ep_params.err_mode        = err_handling_opt.ucp_err_mode;
  //  NCCL_UCX_INFO(NCCL_NET, "Worker: %p", ucp_worker);

  //check if ucp worker exists. The condition can be true iff
  //current process is the root of the tree 
  if (ucp_worker == UCP_WORKER_NONE) {
    ucp_worker_params_t worker_params;
    
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    UCXCHECK(ucp_worker_create(ucp_context, &worker_params, &ucp_worker));   
  }
  UCXCHECK(ucp_ep_create(ucp_worker, &ep_params, &ep));

#if SEND_TYPE == STREAM_SEND
  ucx_request *req;
  int dummy_data = 0xDADADADA;

  req = ucp_stream_send_nb(ep, &dummy_data, sizeof(int), ucp_dt_make_contig(1), send_handler, 0);
  if (UCS_PTR_IS_ERR(req)) {
    fprintf(stderr, "unable to send UCX address message\n");
  } else if (UCS_PTR_STATUS(req) != UCS_OK) {
    wait(ucp_worker, req);
    req->completed = 0; req->release = 0;
    ucp_request_release(req);
  }
#endif
  ucx_send_comm *comm = (ucx_send_comm*)malloc(sizeof(ucx_send_comm));
  comm->worker = ucp_worker;
  comm->ep     = ep;
  *send_comm   = comm;
  ucp_worker = UCP_WORKER_NONE;
  pthread_mutex_unlock(&worker_mutex);
  return ncclSuccess;
}

__hidden ncclResult_t ucx_accept(void* listen_comm, void** recv_comm) {
  //  NCCL_UCX_INFO(NCCL_NET, "ucx_accept");
  *recv_comm = listen_comm;
#if SEND_TYPE == STREAM_SEND
  ucp_worker_h ucp_worker;
  ucp_ep_h ep;
  ucp_stream_poll_ep_t eps;
  ssize_t n_ep;
  int dummy_data;
  size_t length;
  ucx_request *req;

  ucp_worker = (ucp_worker_h) *recv_comm;
  do {
    ucp_worker_progress(ucp_worker);
    n_ep = ucp_stream_worker_poll(ucp_worker, &eps, 1, 0);
    if (UCS_PTR_IS_ERR(n_ep)){
      fprintf(stderr, "error poll stream worker (%u)\n", UCS_PTR_STATUS(n_ep));
      return ncclSystemError;
    }
  } while(n_ep == 0);
  //sanity check
  if (n_ep !=1){
    fprintf(stderr, "too many endpoints for streams (%d)\n", n_ep);
  }
  ep = eps.ep;
  req = ucp_stream_recv_nb(ep, &dummy_data, sizeof(int), ucp_dt_make_contig(1), recv_handler_stream, &length, UCP_STREAM_RECV_FLAG_WAITALL); 
  if (UCS_PTR_IS_ERR(req)) {
    fprintf(stderr, "unable to receive UCX data message (%u)\n", UCS_PTR_STATUS(req));
  } else if (UCS_PTR_STATUS(req) != UCS_OK) {
    wait(ucp_worker, req);
    req->completed = 0; req->release = 0;
    ucp_request_release(req);
  }

  ucx_recv_comm *comm = (ucx_recv_comm*)malloc(sizeof(ucx_recv_comm));
  comm->worker = ucp_worker;
  comm->ep     = ep;
  *recv_comm   = comm;
    
#endif
  return ncclSuccess;
}

__hidden ncclResult_t ucx_regmr(void* comm, void* data, int size, int type, void** mhandle){
  ucp_mem_map_params_t mmap_params;
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                           UCP_MEM_MAP_PARAM_FIELD_FLAGS;
  mmap_params.address    = (uint64_t)data;
  mmap_params.length     = size;
  mmap_params.flags      = UCP_MEM_MAP_FIXED;
  ucp_mem_map(ucp_context, &mmap_params, mhandle);
  return ncclSuccess;
}

__hidden ncclResult_t ucx_deregmr(void* comm, void* mhandle){
  ucp_mem_unmap(ucp_context, mhandle);
  return ncclSuccess;
}

__hidden ncclResult_t ucx_isend(void* send_comm, void* data, int size, void* mhandle, void** request) {
  ucx_request *req;
  ucx_send_comm *comm;

  comm = (ucx_send_comm*) send_comm;
#if SEND_TYPE == TAG_SEND
  req = ucp_tag_send_nb(comm->ep, data, size, ucp_dt_make_contig(1), tag, send_handler);
#elif SEND_TYPE == STREAM_SEND
  req = ucp_stream_send_nb(comm->ep, data, size, ucp_dt_make_contig(1), send_handler, 0);
#endif
  
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

#if SEND_TYPE == TAG_SEND
  ucp_worker = (ucp_worker_h)recv_comm;
  req = ucp_tag_recv_nb(ucp_worker, data, size, ucp_dt_make_contig(1), tag, tag_mask, recv_handler);
#elif SEND_TYPE == STREAM_SEND
  ucx_recv_comm *comm;
  size_t length;

  comm = (ucx_recv_comm*)recv_comm;
  ucp_worker = comm->worker;
  req = ucp_stream_recv_nb(comm->ep, data, size, ucp_dt_make_contig(1), recv_handler_stream, &length, UCP_STREAM_RECV_FLAG_WAITALL);
#endif
  if (UCS_PTR_IS_ERR(req)) {
    NCCL_UCX_WARN("ucx_irecv: unable to receive UCX address message (%s)\n",
            ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  }else if (req == NULL) { //if (UCS_PTR_STATUS(req) == UCS_OK){
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
  return ncclSuccess;
}

__hidden ncclResult_t ucx_test(void* request, int* done, int* size) {
  ucx_request *req = (ucx_request*)request;
  *done = 0;
  int debug = 0;
  while(debug)
    debug = 1;
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

  comm = (ucx_send_comm*) send_comm;
  ucp_ep_destroy(comm->ep);
  ucp_worker_destroy(comm->worker);
  
  //  ucp_cleanup(ucp_context);
  return ncclSuccess;
}

__hidden ncclResult_t ucx_close_recv(void* recv_comm) {
  ucp_worker_h ucp_worker;

  //  ucp_worker  = (ucp_worker_h) recv_comm;
  //  ucp_worker_destroy(ucp_worker);
  return ncclSuccess;
}

__hidden ncclResult_t ucx_close_listen(void* listen_comm) {
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
