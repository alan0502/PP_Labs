#!/usr/bin/env python3
"""
plot_ablation_nvtxsum.py

用於把多個 Nsight Systems nvtx_sum CSV 報告畫成 Ablation Study 時間堆疊圖。
每個 CSV 對應一條柱狀圖。
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============ 設定 ============

# 你的 CSV 路徑（可依實際修改）
csv_files = [
    "/home/pp25/pp25s065/PP2025/HW2/nsys_reports/nosimd/N4_n4_c6/rank_0_nvtx_sum.csv",
    "/home/pp25/pp25s065/PP2025/HW2/nsys_reports/simd/N4_n4_c6/rank_0_nvtx_sum.csv",
    #"/home/pp25/pp25s065/PP2025/HW2/nsys_reports/ilp3/N4_n4_c6/rank_0_nvtx_sum.csv",
    #"/home/pp25/pp25s065/PP2025/HW2/nsys_reports/ilp4/N4_n4_c6/rank_0_nvtx_sum.csv",
]

# 對應的柱狀名稱
labels = [
    "Case 1\n(No SIMD)",
    "Case 2\n(SIMD)",
    #"Case 3\n(ILP = 3)",
    #"Case 4\n(ILP = 4)"
]

# 色彩設定
colors = {
    ":IO": "#8ecae6",      # 淺藍
    ":CPU": "#90be6d",     # 淺綠
    ":Comm": "#f25c54"     # 橘紅
}

# ============ 載入與處理 ============

def read_nvtx_csv(path: str):
    df = pd.read_csv(path)
    df = df[["Range", "Total Time (ns)"]]
    df["Total Time (s)"] = df["Total Time (ns)"] / 1e9
    sums = {r: df.loc[df["Range"] == r, "Total Time (s)"].sum() for r in [":IO", ":CPU", ":Comm"]}
    total = sum(sums.values())
    return sums, total

rows = []
for f, label in zip(csv_files, labels):
    sums, total = read_nvtx_csv(f)
    rows.append({
        "Label": label,
        "IO": sums[":IO"],
        "CPU": sums[":CPU"],
        "Comm": sums[":Comm"],
        "Total": total
    })

df = pd.DataFrame(rows)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(6, 5))

# 調整 bar 寬度
bar_width = 0.2

bars_io = ax.bar(df["Label"], df["IO"], color=colors[":IO"], label="I/O", width=bar_width)
bars_cpu = ax.bar(df["Label"], df["CPU"], bottom=df["IO"], color=colors[":CPU"], label="Compute", width=bar_width)
bottom_comm = df["IO"] + df["CPU"]
bars_comm = ax.bar(df["Label"], df["Comm"], bottom=bottom_comm, color=colors[":Comm"], label="Comm", width=bar_width)

# 標示總時間
for i, total in enumerate(df["Total"]):
    ax.text(i, total + 0.2, f"{total:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Runtime (seconds)", fontsize=12, fontweight="bold")
ax.set_title("Ablation Study: Time Profile", fontsize=14, fontweight="bold", pad=10)
ax.legend(frameon=True, fontsize=10, loc="upper right")
ax.set_xlim(-0.5, len(df) - 0.5)
plt.tight_layout()
plt.savefig("ablation_time_profile.png", dpi=300)
plt.show()
