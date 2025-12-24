#!/bin/bash
# 以 Nsight Systems 包住你的程式；輸出依 Slurm 參數自動分目錄命名
# 用法： nsight_mandel.sh <your exe> <args...>

set -euo pipefail

profile_root="./nsys_reports/nosimd"
# 可能會用到的 Slurm / MPI 變數
NNODES="${SLURM_NNODES:-1}"
NTASKS="${SLURM_NTASKS:-1}"
CPT="${SLURM_CPUS_PER_TASK:-1}"
PRANK="${PMIX_RANK:-0}"     # Open MPI 存在；Pthread 時可能沒有，就設 0

exp_dir="${profile_root}/N${NNODES}_n${NTASKS}_c${CPT}"
mkdir -p "${exp_dir}"

out_file="${exp_dir}/rank_${PRANK}.nsys-rep"

# 追蹤 MPI、UCX、OS runtime、NVTX（依需要增減）
nsys profile \
  -o "${out_file}" \
  --mpi-impl openmpi \
  --trace mpi,ucx,osrt,nvtx \
  "$@"
