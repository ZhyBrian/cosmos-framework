#!/usr/bin/env bash
# Source this file before UMI-FT E0/E1 commands on the A40 host.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source examples/umift/a40_env.sh; do not execute it" >&2
  exit 2
fi

_umift_venv=/data/cosmos_envs/umi_edge_e1_py313
_umift_site="$_umift_venv/lib/python3.13/site-packages"

if [[ ! -x "$_umift_venv/bin/python" ]]; then
  echo "UMI-FT venv is missing: $_umift_venv" >&2
  return 2
fi
case "${CUDA_VISIBLE_DEVICES-}" in
  0|0,1,2,3) ;;
  *)
    echo "set CUDA_VISIBLE_DEVICES explicitly to 0 (attention) or 0,1,2,3 (model/train) before sourcing" >&2
    return 2
    ;;
esac

for _umift_component in curand cudnn cuda_nvrtc; do
  if [[ ! -d "$_umift_site/nvidia/$_umift_component" ]]; then
    echo "missing venv NVIDIA component: $_umift_site/nvidia/$_umift_component" >&2
    return 2
  fi
done

export VIRTUAL_ENV="$_umift_venv"
export PATH="$_umift_venv/bin:$PATH"
export PYTHONNOUSERSITE=1
# Keep this experiment's package, model, compiler and temporary files inside
# the Cosmos namespace on the shared A40 host.
export UV_CACHE_DIR=/data/cosmos_envs/cache/uv
export UV_PYTHON_INSTALL_DIR=/data/cosmos_envs/python
export UV_PYTHON_BIN_DIR=/data/cosmos_envs/bin
export HF_HOME=/data/cosmos_models/cache/huggingface
export TORCH_HOME=/data/cosmos_models/cache/torch
export XDG_CACHE_HOME=/data/cosmos_runs/cache/xdg
export TORCHINDUCTOR_CACHE_DIR=/data/cosmos_runs/cache/torchinductor
export TRITON_CACHE_DIR=/data/cosmos_runs/cache/triton
export CUDA_CACHE_PATH=/data/cosmos_runs/cache/cuda
export TMPDIR=/data/cosmos_runs/tmp
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_BIN_DIR" "$HF_HOME" "$TORCH_HOME" \
  "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$TMPDIR" || return 2
# Transformer Engine consults these component homes before the broken host
# /usr/local/cuda libraries. Keep the override local to this sourced shell.
export CURAND_HOME="$_umift_site/nvidia/curand"
export CUDNN_HOME="$_umift_site/nvidia/cudnn"
export NVRTC_HOME="$_umift_site/nvidia/cuda_nvrtc"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export I4_ATTN_BACKENDS="${I4_ATTN_BACKENDS:-natten}"
unset I4_ATTN_BACKENDS_MULTIDIM

unset _umift_component _umift_site _umift_venv
