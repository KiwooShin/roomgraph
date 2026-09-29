FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc-11 g++-11 git git-lfs ca-certificates curl wget unzip \
    python3 python3-venv python3-pip pkg-config rsync ninja-build \
    libatomic1 libegl1 libgl1 libglu1-mesa libglx0 libgomp1 libsm6 libxi6 \
    libxrandr2 libxt6 libglib2.0-0 libnghttp2-14 libvulkan1 vulkan-tools \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200 \
    && rm -rf /var/lib/apt/lists/*

# Isaac 5.1's bundled NVCC cannot parse glibc's ARM vector-math types.
# Omit only vector-math declarations during CUDA compilation; ordinary C/C++
# keeps the original declarations. This changes the container header only.
RUN python3 -c 'from pathlib import Path; p=Path("/usr/include/aarch64-linux-gnu/bits/math-vector.h"); s=p.read_text(); a="#if __GNUC_PREREQ(9, 0)"; b="#if __GNUC_PREREQ(10, 0) || __glibc_clang_prereq(11, 0)"; assert s.count(a)==1 and s.count(b)==1; p.write_text(s.replace(a, "#if !defined(__CUDACC__) && __GNUC_PREREQ(9, 0)").replace(b, "#if !defined(__CUDACC__) && (__GNUC_PREREQ(10, 0) || __glibc_clang_prereq(11, 0))"))'

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN userdel ubuntu && groupadd --non-unique --gid ${GROUP_ID} renderer \
    && useradd --non-unique --uid ${USER_ID} --gid ${GROUP_ID} --home-dir /home/ubuntu --create-home renderer
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
USER renderer
WORKDIR /isaac-sim
