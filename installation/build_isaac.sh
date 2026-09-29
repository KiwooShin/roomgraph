#!/usr/bin/env bash
# Build the pinned native ARM64 release in a disposable compiler environment.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
install_dir=${ROOMGRAPH_INSTALL_DIR:-${HOME}/.local/opt/indoor-structure}
source_dir=${install_dir}/IsaacSim-5.1.0
mkdir -p "${install_dir}/container-home" "${project_dir}/artifacts"
if [[ ! -d "${source_dir}/.git" ]]; then
    git clone --depth 1 --branch v5.1.0 https://github.com/isaac-sim/IsaacSim.git "${source_dir}"
fi
test "$(git -C "${source_dir}" rev-parse HEAD)" = 47d886f2858d1ceed556b21c88927aa67bc81c12
python3 "${project_dir}/installation/patch_isaac.py" "${source_dir}"
docker build --build-arg USER_ID="$(id -u)" --build-arg GROUP_ID="$(id -g)" \
    -t roomgraph/isaac-build:5.1.0 "${project_dir}/installation"
docker run --rm -i --name roomgraph-isaac-build \
    -v "${source_dir}:/isaac-sim" \
    -v "${install_dir}/container-home:/home/ubuntu" \
    roomgraph/isaac-build:5.1.0 bash -c \
    'git lfs install --local && git lfs pull && ./build.sh --release'
