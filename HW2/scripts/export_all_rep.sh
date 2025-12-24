#!/bin/bash
# 批次匯出所有 Nsight Systems 報告為 CSV
# 用法：
#   bash export_all_nsys_csv.sh
# 預設搜尋 ./nsys_reports/mandelbrot 下的所有 rank_*.nsys-rep

set -euo pipefail

ROOT="./nsys_reports/mandelbrot"
SEARCH_PATTERN="rank_*.nsys-rep"

echo "[INFO] Start exporting all .nsys-rep files under $ROOT ..."
echo

find "$ROOT" -type f -name "$SEARCH_PATTERN" | while read -r repfile; do
    outdir="$(dirname "$repfile")"
    basename="$(basename "$repfile" .nsys-rep)"
    outfile="$outdir/$basename"
    echo "[RUN] nsys export -t csv -o $outfile $repfile"
    nsys export -t csv -o "$outfile" "$repfile"
    echo "[OK]  Exported -> ${outfile}_nvtx_events.csv"
    echo
done

echo "[DONE] All exports completed."
