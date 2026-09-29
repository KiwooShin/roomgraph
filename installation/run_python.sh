#!/usr/bin/env bash
# Paths passed to Python should be relative to the project directory.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
install_dir=${ROOMGRAPH_INSTALL_DIR:-${HOME}/.local/opt/indoor-structure}
docker_args=()
if [[ -n "${ROOMGRAPH_CONTAINER_CIDFILE:-}" ]]; then
    docker_args+=(--cidfile "${ROOMGRAPH_CONTAINER_CIDFILE}")
fi
docker run "${docker_args[@]}" --rm --gpus all --shm-size=4g \
    -v "${install_dir}/IsaacSim-5.1.0:/isaac-sim" \
    -v "${install_dir}/container-home:/home/ubuntu" \
    -v "${project_dir}:/workspace" -w /workspace \
    -e PYTHONPATH=/workspace/src \
    -e LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
    roomgraph/isaac-build:5.1.0 \
    /isaac-sim/_build/linux-aarch64/release/python.sh "$@"
