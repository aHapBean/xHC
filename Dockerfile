FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:e071e85c52ad91fc9ea24158ff5330876b2d1a5c4ac83ccc6066976835873c01

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-venv git build-essential ninja-build cmake \
    libnuma-dev libibverbs-dev librdmacm-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/xhc
ENV PATH=/opt/xhc/bin:$PATH CUDA_HOME=/usr/local/cuda \
    CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
ARG TORCH_CUDA_ARCH_LIST=10.0
ARG NVTE_CUDA_ARCHS=100
ARG MAX_JOBS=8
WORKDIR /opt/xHC
COPY requirements.txt constraints-cu130.txt ./
COPY scripts/install_environment.sh ./scripts/install_environment.sh
COPY Megatron-LM ./Megatron-LM
RUN bash scripts/install_environment.sh
ARG WITH_HYBRID_EP=1
COPY scripts/install_hybridep.sh ./scripts/install_hybridep.sh
RUN if [ "$WITH_HYBRID_EP" = 1 ]; then bash scripts/install_hybridep.sh; fi
COPY configs ./configs
COPY scripts ./scripts
CMD ["bash"]
