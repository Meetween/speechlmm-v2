FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

WORKDIR /workspace

# update apt
RUN apt update

# install curl and git
RUN apt install -y curl git

# install RDMA/InfiniBand userspace libraries matching host MLNX_OFED 23.10
# Download debs directly from MLNX OFED repo (GPG key no longer available)
RUN apt install -y wget && \
    mkdir -p /tmp/mlnx-debs && cd /tmp/mlnx-debs && \
    MLNX_REPO="https://linux.mellanox.com/public/repo/mlnx_ofed/23.10-0.5.5.0/ubuntu22.04/x86_64" && \
    wget -q "${MLNX_REPO}/rdma-core_2307mlnx47-1.2310055_amd64.deb" && \
    wget -q "${MLNX_REPO}/libibverbs1_2307mlnx47-1.2310055_amd64.deb" && \
    wget -q "${MLNX_REPO}/libibverbs-dev_2307mlnx47-1.2310055_amd64.deb" && \
    wget -q "${MLNX_REPO}/ibverbs-providers_2307mlnx47-1.2310055_amd64.deb" && \
    wget -q "${MLNX_REPO}/ibverbs-utils_2307mlnx47-1.2310055_amd64.deb" && \
    wget -q "${MLNX_REPO}/librdmacm1_2307mlnx47-1.2310055_amd64.deb" && \
    wget -q "${MLNX_REPO}/librdmacm-dev_2307mlnx47-1.2310055_amd64.deb" && \
    wget -q "${MLNX_REPO}/infiniband-diags_2307mlnx47-1.2310055_amd64.deb" && \
    wget -q "${MLNX_REPO}/libibumad3_2307mlnx47-1.2310055_amd64.deb" && \
    wget -q "${MLNX_REPO}/libibnetdisc5_2307mlnx47-1.2310055_amd64.deb" && \
    apt install -y libnl-3-200 libnl-route-3-200 && \
    dpkg -i --force-overwrite *.deb || apt-get install -f -y && \
    rm -rf /tmp/mlnx-debs /var/lib/apt/lists/*

# install modules
ENV TZ=Europe/Rome
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y environment-modules

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# install uv (to /usr/local/bin so it's accessible to non-root users, e.g. in Singularity)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN UV_INSTALL_DIR=/usr/local/bin sh /uv-installer.sh && rm /uv-installer.sh

# store uv-managed Python in a globally accessible path (needed for Singularity, which runs as non-root)
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python

CMD ["bash"]
