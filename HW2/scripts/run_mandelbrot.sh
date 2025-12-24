#!/bin/bash
# 用法：
#   bash run_mandelbrot.sh
# Mandelbrot set 實驗腳本
# 自動呼叫 nsight_mandel.sh 產生分析檔

set -euo pipefail

###############################
# 基本設定（按需修改）
###############################
# 節點數
N=4

# 實驗組合（MPI ranks 與每 rank threads）
# Pthread 版請把 PROCS=(1)
PROCS=(4)
THREADS=(6)

# 是否開啟 Nsight Systems
PROFILE=1            # 1=開；0=關
NSYS_WRAPPER="scripts/nsight.sh"

# 你的可執行檔
EXE="src/hw2b_nosimd"

###############################
# Mandelbrot 測資參數
# 來自你提供的 testcase
###############################
ITER=174170376
X0=-0.7894722222222222
X1=-0.7825277777777778
Y0=0.145046875
Y1=0.148953125
W=2549
H=1439

# 輸出圖片
OUT_BASE="out"

###############################
# OpenMP 環境設定（若有）
###############################
export OMP_PLACES=cores
export OMP_PROC_BIND=close

###############################
# 執行
###############################
echo "Executable : $EXE"
echo "BBox/Size  : [$X0,$X1] x [$Y0,$Y1], ${W}x${H}, iter=$ITER"
echo "Nodes      : $N"
echo "Profile    : $PROFILE"
echo

for p in "${PROCS[@]}"; do
  for t in "${THREADS[@]}"; do
    tag="N${N}_n${p}_c${t}_it${ITER}_${W}x${H}"
    OUT="${OUT_BASE}_${tag}.png"

    echo "==== Running: -N $N -n $p -c $t -> ${OUT} ===="
    export OMP_NUM_THREADS="$t"

    if [[ "$PROFILE" -eq 1 ]]; then
      srun -N "$N" -n "$p" -c "$t" "$NSYS_WRAPPER" \
        "$EXE" "$OUT" "$ITER" "$X0" "$X1" "$Y0" "$Y1" "$W" "$H"
    else
      srun -N "$N" -n "$p" -c "$t" \
        "$EXE" "$OUT" "$ITER" "$X0" "$X1" "$Y0" "$Y1" "$W" "$H"
    fi

    echo
  done
done
