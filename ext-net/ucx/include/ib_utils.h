/*************************************************************************
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef IB_UTILS_H_
#define IB_UTILS_H_
#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glob.h>


#define MAX_STR_LEN 128
#define PREF "/sys/class/net/"
#define SUFF "/device/resource"

static int cmp_files(char *f1, char *f2) {
    int answer = 0;
    FILE *fp1, *fp2;

    if ((fp1 = fopen(f1, "r")) == NULL)
        goto out;
    else if ((fp2 = fopen(f2, "r")) == NULL)
        goto close;

    int ch1 = getc(fp1);
    int ch2 = getc(fp2);

    while((ch1 != EOF) && (ch2 != EOF) && (ch1 == ch2)){
        ch1 = getc(fp1);
        ch2 = getc(fp2) ;
    }

    if (ch1 == ch2)
        answer = 1;

    fclose(fp2);
close:
    fclose(fp1);
out:
    return answer;
}

static int port_from_file(char *port_file) {
    char buf1[MAX_STR_LEN], buf2[MAX_STR_LEN];
    FILE *fp;
    int res = -1;

    if ((fp = fopen(port_file, "r")) == NULL)
        return -1;

    if (fgets(buf1, MAX_STR_LEN - 1, fp) == NULL)
        goto out;

    int len = strlen(buf1) - 2;
    strncpy(buf2, buf1 + 2, len);
    buf2[len] = 0;
    res = atoi(buf2);

out:
    fclose(fp);
    return res;
}

static int dev2if(char *dev_name, int port, char *if_name) {
    char dev_file[MAX_STR_LEN], port_file[MAX_STR_LEN], net_file[MAX_STR_LEN];
    char glob_path[MAX_STR_LEN];
    glob_t glob_el = {0,};
    int found = 0;

    char *env = getenv("NCCL_UCX_NET_FILE_PREFIX");
    if (env != NULL) {
        sprintf(glob_path, PREF"%s*", env);
    } else {
        sprintf(glob_path, PREF"*");
    }

    sprintf(dev_file, "/sys/class/infiniband/%s"SUFF, dev_name);

    glob(glob_path, 0, 0, &glob_el);
    char **p = glob_el.gl_pathv;

    if (glob_el.gl_pathc >= 1) {
        for(int i = 0; i < glob_el.gl_pathc; i++, p++){
            sprintf(port_file, "%s/dev_id", *p);
            sprintf(net_file,  "%s"SUFF,    *p);
            if(cmp_files(net_file, dev_file)  &&
               port_from_file(port_file) == port - 1)
            {
                found = 1;
                break;
            }
        }
    }

    globfree(&glob_el);

    if(found){
        int len = strlen(net_file) - strlen(PREF) - strlen(SUFF);
        strncpy(if_name, net_file + strlen(PREF), len);
        if_name[len] = 0;
    }
    else{
        strcpy(if_name, "");
    }

    return found;
}

static int get_ipoib_ip(char *ifname, struct sockaddr_storage *addr){
    struct ifaddrs *ifaddr, *ifa;
    int family, n, is_ipv4 = 0;
    char host[1025];
    const char* host_ptr;

    int rval, ret = 0, is_up;

    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        return ret;
    }

    for (ifa = ifaddr, n = 0; ifa != NULL; ifa=ifa->ifa_next, n++) {
        if (ifa->ifa_addr == NULL)
            continue;

        family = ifa->ifa_addr->sa_family;
        if (family != AF_INET && family != AF_INET6)
            continue;

        is_up = (ifa->ifa_flags & IFF_UP) == IFF_UP;
        is_ipv4 = (family == AF_INET) ? 1 : 0;

        if (is_up && !strncmp(ifa->ifa_name, ifname, strlen(ifname)) ) {
            if (is_ipv4) {
                memcpy((struct sockaddr_in *) addr,
                       (struct sockaddr_in *) ifa->ifa_addr,
                       sizeof(struct sockaddr_in));
            }
            else {
                memcpy((struct sockaddr_in6 *) addr,
                       (struct sockaddr_in6 *) ifa->ifa_addr,
                       sizeof(struct sockaddr_in6));
            }
            ret = 1;
            break;
        }
    }

    freeifaddrs(ifaddr);
    return ret;
}

#endif