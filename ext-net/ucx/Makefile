#
# Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.PHONY : test build
default: build

BUILDDIR=build
NCCL_HOME ?= ../../build/
CUDA_HOME ?= /usr/local/cuda
UCX_HOME ?= /opt/ucx

INC:= -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -I$(UCX_HOME)/include -Iinclude/
CFLAGS:=-fPIC $(INC) -std=c99 -D_GNU_SOURCE -g -O0
PLUGIN_SO:=$(BUILDDIR)/lib/libnccl-net.so
FILES:=nccl_ucx.c utils.c
OBJS:=$(FILES:%.c=$(BUILDDIR)/obj/%.o)
TEST:=$(BUILDDIR)/bin/test

$(BUILDDIR)/obj/%.o: %.c
	mkdir -p $(BUILDDIR)/obj
	$(CC) $(CFLAGS) -c $< -o $@

$(PLUGIN_SO): $(OBJS)
	mkdir -p $(BUILDDIR)/lib
	$(CC) -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $(OBJS) -L$(CUDA_HOME)/lib64 -lcudart -L$(UCX_HOME)/lib -lucp -luct -lucs -lucm

build: $(PLUGIN_SO)
test:  $(TEST)

$(TEST): test.c $(PLUGIN_SO)
	mkdir -p $(BUILDDIR)/bin
	LD_LIBRARY_PATH=$(UCX_HOME)/lib:$LD_LIBRARY_PATH $(CC) $(CFLAGS) test.c -o $@ -L$(BUILDDIR)/lib -lnccl-net
	LD_LIBRARY_PATH=$(UCX_HOME)/lib:$LD_LIBRARY_PATH $(TEST)

clean:
	rm -f $(PLUGIN_SO) $(OBJS) $(TEST)
