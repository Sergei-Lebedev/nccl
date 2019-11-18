
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

struct ucx_listen_handle
{
  union socketAddress connectAddr;
};
typedef struct ucx_listen_handle ucx_listen_handle;

struct ucx_listen_comm
{
  int fd;
  int dev;
};
typedef struct ucx_listen_comm ucx_listen_comm;

struct connect_msg
{
  size_t addr_len;
};
typedef struct connect_msg connect_msg;

struct ucx_request
{
  int completed;
  ucp_worker_h worker;
};
typedef struct ucx_request ucx_request;

struct nccl_ucx_worker
{
  ucp_context_h ctx;
  ucp_worker_h worker;
  int count;
};
static struct nccl_ucx_worker workers[MAX_IB_DEVS];

#define MAX_REQUESTS 128
struct nccl_ucx_send_fifo
{
  uint32_t ready;
};
typedef struct nccl_ucx_send_fifo nccl_ucx_send_fifo;

struct ucx_send_comm
{
  ucp_context_h ctx;
  ucp_worker_h worker;
  ucp_ep_h ep;
  int fd;
  int ready;
  ucp_tag_t tag;
  nccl_ucx_send_fifo fifo[MAX_REQUESTS];
  uint32_t fifo_head;
  ucp_mem_h fifo_memh;
  void *rkey_buf;
  size_t rkey_buf_size;
};
typedef struct ucx_send_comm ucx_send_comm;

struct nccl_ucx_rem_fifo
{
  nccl_ucx_send_fifo elems[MAX_REQUESTS];
  uint64_t addr;
  uint32_t tail;
};
typedef struct nccl_ucx_rem_fifo nccl_ucx_rem_fifo;

struct ucx_recv_comm
{
  ucp_context_h ctx;
  ucp_worker_h worker;
  int fd;
  ucp_tag_t tag;
  nccl_ucx_rem_fifo rem_fifo;
  void *rkey_buf;
  size_t rkey_buf_size;
  ucp_ep_h ep;
  ucp_rkey_h rkey;
  connect_msg *msg;
  ucx_request *connect_req;
  int ready;
};

typedef struct ucx_recv_comm ucx_recv_comm;

static void request_init(void *request)
{
  struct ucx_request *req = (struct ucx_request *)request;
  req->completed = 0;
}

static void send_handler(void *request, ucs_status_t status)
{
  struct ucx_request *req = (struct ucx_request *)request;
  req->completed = 1;
  //   printf("[0x%x] send handler called with status %d (%s)\n", (unsigned int)pthread_self(), status, ucs_status_string(status));
}

static void failure_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
  ucs_status_t *arg_status = (ucs_status_t *)arg;
  //    printf("[0x%x] failure handler called with status %d (%s)\n",(unsigned int)pthread_self(), status, ucs_status_string(status));
  *arg_status = status;
}

static void recv_handler(void *request, ucs_status_t status, ucp_tag_recv_info_t *info)
{
  struct ucx_request *req = (struct ucx_request *)request;
  req->completed = 1;
  //printf("[0x%x] receive handler called with status %d (%s), length %lu\n", (unsigned int)pthread_self(), status, ucs_status_string(status), info->length);
}

static union socketAddress nccl_ucx_if_addr;
static char if_name[MAX_IF_NAME_SIZE];

static ncclResult_t get_socket_addr(union socketAddress *addr)
{
  memcpy(addr, &nccl_ucx_if_addr, sizeof(*addr));
  return ncclSuccess;
}

static ncclResult_t ucx_init_context(ucp_context_h *ctx, int dev)
{
  ucp_params_t ucp_params;
  ucp_config_t *config;

  char ucx_prefix[PATH_MAX];  //DEV_####
  char ucx_env_var[PATH_MAX]; //UCX_DEV_####_NET_DEVICES
  char ucx_env_val[PATH_MAX]; //e.g. mlx5_0:1

  snprintf(ucx_prefix, PATH_MAX, "DEV_%d", dev);
  snprintf(ucx_env_var, PATH_MAX, "UCX_%s_NET_DEVICES", ucx_prefix);
  snprintf(ucx_env_val, PATH_MAX, "%s:%d", ncclIbDevs[dev].devName, ncclIbDevs[dev].port);
  setenv(ucx_env_var, ucx_env_val, 0);
  UCXCHECK(ucp_config_read(ucx_prefix, NULL, &config));

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

static ncclResult_t ucx_init_worker(ucp_context_h ctx, ucp_worker_h *worker)
{
  ucp_worker_params_t worker_params;
  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  UCXCHECK(ucp_worker_create(ctx, &worker_params, worker));
}

ncclResult_t ucx_worker_get_netaddress(ucp_worker_h worker, ucp_address_t **address, size_t *address_length)
{
  ucp_worker_attr_t attr;
  attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS |
                    UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
  attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;
  UCXCHECK(ucp_worker_query(worker, &attr));
  *address = malloc(attr.address_length);
  memcpy(*address, attr.address, attr.address_length);
  *address_length = attr.address_length;
  free(attr.address);
}

static ncclResult_t ucx_get_ctx_and_worker(int dev, ucp_context_h *ctx, ucp_worker_h *worker, ucp_tag_t *newtag)
{
  pthread_mutex_lock(&ncclIbLock);
  if (ncclNIbDevs < dev)
  {
    NCCL_UCX_WARN("Device index is too large");
    return ncclSystemError;
  }
  if (workers[dev].count == 0)
  {
    ucx_init_context(&workers[dev].ctx, dev);
    ucx_init_worker(workers[dev].ctx, &workers[dev].worker);
  }
  *ctx = workers[dev].ctx;
  *worker = workers[dev].worker;
  if (newtag != NULL)
  {
    *newtag = tag + 2 * workers[dev].count;
  }
  workers[dev].count++;
  pthread_mutex_unlock(&ncclIbLock);
  return ncclSuccess;
}

ncclResult_t ucx_init(ncclDebugLogger_t logFunction)
{
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

      for (int d = 0; d < nIbDevs; d++)
      {
        struct ibv_context *context;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL)
        {
          NCCL_UCX_WARN("NET/UCX : Unable to open device %s", devices[d]->name);
          continue;
        }
        int found = 0;
        struct ibv_device_attr devAttr;
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr))
        {
          NCCL_UCX_WARN("NET/UCX : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context))
          {
            return ncclInternalError;
          }
          continue;
        }
        for (int port = 1; port <= devAttr.phys_port_cnt; port++)
        {
          struct ibv_port_attr portAttr;
          if (ncclSuccess != wrap_ibv_query_port(context, port, &portAttr))
          {
            WARN("NET/UCX : Unable to query port %d", port);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE)
            continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET)
            continue;

          // check against user specified HCAs/ports
          if (!(matchIfList(devices[d]->name, port, userIfs, nUserIfs) ^ searchNot))
          {
            continue;
          }
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].context = context;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          //ucx_init_context(&workers[ncclNIbDevs].ctx, ncclNIbDevs);
          //ucx_init_worker(workers[ncclNIbDevs].ctx, &workers[ncclNIbDevs].worker);
          workers[ncclNIbDevs].count = 0;
          ncclNIbDevs++;
          found++;
          //pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, context);
        }
        if (found == 0 && ncclSuccess != wrap_ibv_close_device(context))
        {
          return ncclInternalError;
        }
      }
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices)))
      {
        return ncclInternalError;
      };
    }
    if (ncclNIbDevs == 0)
    {
      INFO(NCCL_INIT | NCCL_NET, "NET/UCX : No device found.");
    }
    else
    {
      char line[1024];
      line[0] = '\0';
      for (int d = 0; d < ncclNIbDevs; d++)
      {
        snprintf(line + strlen(line), 1023 - strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
                 ncclIbDevs[d].port, ncclIbDevs[d].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
      }
      line[1023] = '\0';
      char addrline[1024];
      INFO(NCCL_INIT | NCCL_NET, "NET/UCX : Using%s ; OOB %s", line, if_name);
    }
    pthread_mutex_unlock(&ncclIbLock);
  }
  return ncclSuccess;
}

ncclResult_t ucx_listen(int dev, void *handle, void **listen_comm)
{
  ucx_listen_handle *my_handle;
  ucx_listen_comm *comm;

  //allocate listen comm which contains ucp_worker and socket address to exchange
  comm = malloc(sizeof(ucx_listen_comm));
  memset(comm, 0, sizeof(ucx_listen_comm));

  static_assert(sizeof(ucx_listen_handle) < NCCL_NET_HANDLE_MAXSIZE, "ucx listen handle size too large");
  my_handle = (ucx_listen_handle *)handle;
  comm->dev = dev;
  NCCLCHECK(get_socket_addr(&(my_handle->connectAddr)));
  NCCLCHECK(createListenSocket(&comm->fd, &my_handle->connectAddr));
  *listen_comm = comm;

  return ncclSuccess;
}

ncclResult_t ucx_connect(int dev, void *handle, void **send_comm)
{
  ucp_worker_params_t worker_params;

  ucx_listen_handle *recv_handle = (ucx_listen_handle *)handle;
  ucx_send_comm *comm = (ucx_send_comm *)malloc(sizeof(ucx_send_comm));
  memset(comm, 0, sizeof(ucx_send_comm));

  NCCLCHECK(connectAddress(&comm->fd, &recv_handle->connectAddr));
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker, &comm->tag));
  //ucx_init_context(&comm->ctx, dev);
  //ucx_init_worker(comm->ctx, &comm->worker); comm->tag = tag;

  ucp_mem_map_params_t mmap_params;
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  mmap_params.address = (void *)comm->fifo;
  mmap_params.length = sizeof(nccl_ucx_send_fifo) * MAX_REQUESTS;
  ucp_mem_map(comm->ctx, &mmap_params, &comm->fifo_memh);
  ucp_rkey_pack(comm->ctx, comm->fifo_memh, &comm->rkey_buf, &comm->rkey_buf_size);
  uint64_t fifo_adr = (uint64_t)comm->fifo;

  NCCLCHECK(socketSend(comm->fd, &comm->rkey_buf_size, sizeof(size_t)));
  NCCLCHECK(socketSend(comm->fd, comm->rkey_buf, comm->rkey_buf_size));
  NCCLCHECK(socketSend(comm->fd, &fifo_adr, sizeof(uint64_t)));

  ucp_address_t *my_addr;
  size_t local_addr_len;

  //  UCXCHECK(ucp_worker_get_address(comm->worker, &my_addr, &local_addr_len));
  NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));
  NCCL_UCX_INFO(NCCL_NET, "Worker address length: %zu", local_addr_len);

  NCCLCHECK(socketSend(comm->fd, &local_addr_len, sizeof(size_t)));
  NCCLCHECK(socketSend(comm->fd, my_addr, local_addr_len));
  NCCLCHECK(socketSend(comm->fd, &comm->tag, sizeof(ucp_tag_t)));

  free(my_addr);
  //  ucp_worker_release_address(comm->worker, my_addr);

  comm->ready = 0;
  comm->ep = NULL;
  *send_comm = comm;
  return ncclSuccess;
}

ncclResult_t ucx_accept(void *listen_comm, void **recv_comm)
{
  ucx_recv_comm *r_comm = (ucx_recv_comm *)malloc(sizeof(ucx_recv_comm));
  ucx_listen_comm *l_comm = (ucx_listen_comm *)listen_comm;

  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(l_comm->fd, (struct sockaddr_in *)&sockaddr, &socklen), "accept", r_comm->fd);

  NCCLCHECK(ucx_get_ctx_and_worker(l_comm->dev, &r_comm->ctx, &r_comm->worker, NULL));
  //ucx_init_context(&r_comm->ctx, l_comm->dev);
  //ucx_init_worker(r_comm->ctx, &r_comm->worker); r_comm->tag = tag + rand();

  NCCLCHECK(socketReceive(r_comm->fd, &r_comm->rkey_buf_size, sizeof(size_t)));
  r_comm->rkey_buf = malloc(r_comm->rkey_buf_size);
  NCCLCHECK(socketReceive(r_comm->fd, r_comm->rkey_buf, r_comm->rkey_buf_size));
  NCCLCHECK(socketReceive(r_comm->fd, &r_comm->rem_fifo.addr, sizeof(uint64_t)));

  size_t peer_addr_len;
  ucp_address_t *peer_addr;
  ucp_ep_h ep;
  ucp_ep_params_t ep_params;
  NCCLCHECK(socketReceive(r_comm->fd, &peer_addr_len, sizeof(size_t)));
  peer_addr = malloc(peer_addr_len);
  NCCLCHECK(socketReceive(r_comm->fd, peer_addr, peer_addr_len));
  NCCLCHECK(socketReceive(r_comm->fd, &r_comm->tag, sizeof(ucp_tag_t)));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS; //|
  //                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = peer_addr;
  //  ep_params.err_mode        = err_handling_opt.ucp_err_mode;
  //  NCCL_UCX_INFO(NCCL_NET, "Worker: %p", ucp_worker);
  UCXCHECK(ucp_ep_create(r_comm->worker, &ep_params, &r_comm->ep));
  UCXCHECK(ucp_ep_rkey_unpack(r_comm->ep, r_comm->rkey_buf, &r_comm->rkey));

  free(peer_addr);
  r_comm->ready = 0;
  r_comm->connect_req = NULL;
  *recv_comm = r_comm;

  return ncclSuccess;
}

ncclResult_t ucx_send_check(ucx_send_comm *comm)
{
  ucp_tag_message_h msg_tag;
  ucp_tag_recv_info_t info_tag;
  ucx_request *req;
  connect_msg *msg;
  ucp_address_t *peer_addr;
  ucp_ep_h ep;
  ucp_ep_params_t ep_params;

  ucp_worker_progress(comm->worker);

  msg_tag = ucp_tag_probe_nb(comm->worker, comm->tag + 1, tag_mask, 1, &info_tag);
  if (msg_tag == NULL)
  {
    return ncclSuccess;
  }
  msg = malloc(info_tag.length);
  req = ucp_tag_msg_recv_nb(comm->worker, msg, info_tag.length, ucp_dt_make_contig(1), msg_tag, recv_handler);
  if (UCS_PTR_IS_ERR(req))
  {
    NCCL_UCX_WARN("Unable to receive connect msg (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
  }
  else
  {
    while (req->completed == 0)
    {
      ucp_worker_progress(comm->worker);
    }
    req->completed = 0;
    ucp_request_release(req);
  }
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS; //|
  //                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = msg + 1;
  //  ep_params.err_mode        = err_handling_opt.ucp_err_mode;
  UCXCHECK(ucp_ep_create(comm->worker, &ep_params, &ep));
  comm->ep = ep;
  comm->ready = 1;

  free(msg);
  return ncclSuccess;
}

ncclResult_t ucx_recv_check(ucx_recv_comm *comm)
{
  if (comm->connect_req == NULL){
    ucp_address_t *my_addr;
    size_t local_addr_len;
    NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));
    size_t msg_len = sizeof(connect_msg) + local_addr_len;
    comm->msg = calloc(1, msg_len);
    comm->msg->addr_len = local_addr_len;
    memcpy(comm->msg + 1, my_addr, local_addr_len);
    comm->connect_req = ucp_tag_send_nb(comm->ep, comm->msg, msg_len, ucp_dt_make_contig(1), comm->tag + 1, send_handler);
    if (UCS_PTR_IS_ERR(comm->connect_req))
    {
      NCCL_UCX_WARN("Unable to send connect message");
      return ncclSystemError;
    } else if (comm->connect_req == NULL){
      comm->ready = 1;
    }
    free(my_addr);
  }
  else{
    if (comm->connect_req->completed == 1) {
      comm->ready = 1;
      comm->connect_req->completed = 0;
      ucp_request_release(comm->connect_req);
    }
    else {
      ucp_worker_progress(comm->worker);
    }
  }
}

ncclResult_t ucx_isend(void *send_comm, void *data, int size, void *mhandle, void **request)
{
  ucx_request *req;
  ucx_send_comm *comm = (ucx_send_comm *)send_comm;

  if (comm->ready == 0) { ucx_send_check(comm); }
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }

  volatile nccl_ucx_send_fifo *slot = comm->fifo + (comm->fifo_head % MAX_REQUESTS);
  volatile uint32_t *ready_ptr = &slot->ready;
  if (*ready_ptr == 0)
  {
    *request = NULL;
    return ncclSuccess;
  }
  req = ucp_tag_send_nb(comm->ep, data, size, ucp_dt_make_contig(1), comm->tag, send_handler);
  if (UCS_PTR_IS_ERR(req))
  {
    NCCL_UCX_WARN("ucx_isend: unable to send message (%s)\n", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  }
  else if (req != NULL)
  {
    ucp_worker_progress(comm->worker);
    req->worker = comm->worker;
  }
  slot->ready = 0;
  comm->fifo_head++;
  *request = req ? req : 1;
  return ncclSuccess;
}

ncclResult_t ucx_irecv(void *recv_comm, void *data, int size, void *mhandle, void **request)
{
  ucx_request *req;
  ucx_recv_comm *comm = (ucx_recv_comm *)recv_comm;

  if (comm->ready == 0) { ucx_recv_check(comm); }
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }

  req = ucp_tag_recv_nb(comm->worker, data, size, ucp_dt_make_contig(1), comm->tag, tag_mask, recv_handler);
  if (UCS_PTR_IS_ERR(req))
  {
    NCCL_UCX_WARN("ucx_irecv: unable to receive message (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  }
  else if (req != NULL)
  {
    ucp_worker_progress(comm->worker);
    req->worker = comm->worker;
  }
  nccl_ucx_send_fifo *local_elem = comm->rem_fifo.elems + (comm->rem_fifo.tail % MAX_REQUESTS);
  local_elem->ready = 1;
  ucp_put_nbi(comm->ep, local_elem, sizeof(nccl_ucx_send_fifo), comm->rem_fifo.addr + (comm->rem_fifo.tail % MAX_REQUESTS) * sizeof(struct nccl_ucx_send_fifo), comm->rkey);
  comm->rem_fifo.tail++;
  *request = req ? req : 1;
  return ncclSuccess;
}

ncclResult_t ucx_test(void *request, int *done, int *size)
{
  ucx_request *req = (ucx_request *)request;
  *done = 0;
  // we don't set size cause we don't use it later in ucx_flush
  //  if(size) *size = 0;
  if (request == 1)
  {
    *done = 1;
    return ncclSuccess;
  }
  if (req->completed == 1)
  {
    *done = 1;
    req->completed = 0;
    ucp_request_release(req);
  }
  else
  {
    ucp_worker_progress(req->worker);
  }
  return ncclSuccess;
}

ncclResult_t ucx_close_send(void *send_comm)
{
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

ncclResult_t ucx_close_recv(void *recv_comm)
{
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

ncclResult_t ucx_close_listen(void *listen_comm)
{
  ucx_listen_comm *comm = (ucx_listen_comm *)listen_comm;
  if (comm)
  {
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
    ucx_close_listen};
