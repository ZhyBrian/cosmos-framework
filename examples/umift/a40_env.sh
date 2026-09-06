#!/usr/bin/env bash
# Source this file before UMI-FT E0/E1 commands on the A40 host.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source examples/umift/a40_env.sh; do not execute it" >&2
  exit 2
fi

_umift_conda_root=/data/miniconda3
_umift_env=/data/miniconda3/envs/cosmos_edge_e1
_umift_conda_sh="$_umift_conda_root/etc/profile.d/conda.sh"

if [[ ! -f "$_umift_conda_sh" ]]; then
  echo "Miniconda activation script is missing: $_umift_conda_sh" >&2
  return 2
fi
case "${CUDA_VISIBLE_DEVICES-}" in
  0|0,1,2,3) ;;
  *)
    echo "set CUDA_VISIBLE_DEVICES explicitly to 0 (attention) or 0,1,2,3 (model/train) before sourcing" >&2
    return 2
    ;;
esac
source "$_umift_conda_sh" || return 2
conda activate "$_umift_env" || return 2
hash -r
if [[ "${CONDA_PREFIX-}" != "$_umift_env" ]]; then
  echo "wrong Conda environment active: expected $_umift_env, got ${CONDA_PREFIX-<unset>}" >&2
  return 2
fi
_umift_python="$(command -v python)"
_umift_python_real="$(python -c 'import os, sys; print(os.path.realpath(sys.executable))')" || return 2
if [[ "$_umift_python" != "$_umift_env/bin/python" || "$_umift_python_real" != "$_umift_env"/* ]]; then
  echo "Python is not provided by $_umift_env: command=$_umift_python executable=$_umift_python_real" >&2
  return 2
fi

_umift_site="$_umift_env/lib/python3.13/site-packages"
for _umift_component in curand cudnn cuda_nvrtc; do
  if [[ ! -d "$_umift_site/nvidia/$_umift_component" ]]; then
    echo "missing Conda NVIDIA component: $_umift_site/nvidia/$_umift_component" >&2
    return 2
  fi
done

export PYTHONNOUSERSITE=1
# Keep this experiment's package, model, compiler and temporary files inside
# the Cosmos namespace on the shared A40 host.
export CONDA_PKGS_DIRS=/data/cosmos_conda/pkgs
export PIP_CACHE_DIR=/data/cosmos_conda/cache/pip
export UV_CACHE_DIR=/data/cosmos_conda/cache/uv
export HF_HOME=/data/cosmos_models/cache/huggingface
export TORCH_HOME=/data/cosmos_models/cache/torch
export EDGE_HF_SNAPSHOT_PATH="${EDGE_HF_SNAPSHOT_PATH:-/data/cosmos_models/Cosmos3-Edge/snapshots/a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/data/cosmos_models/Wan2.2-VAE-921dbaf/Wan2.2_VAE.pth}"
export XDG_CACHE_HOME=/data/cosmos_runs/cache/xdg
export TORCHINDUCTOR_CACHE_DIR=/data/cosmos_runs/cache/torchinductor
export TRITON_CACHE_DIR=/data/cosmos_runs/cache/triton
export CUDA_CACHE_PATH=/data/cosmos_runs/cache/cuda
export TMPDIR=/data/cosmos_runs/tmp
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
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

unset _umift_component _umift_conda_root _umift_conda_sh _umift_env _umift_python _umift_python_real _umift_site
