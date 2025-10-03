import matplotlib.pyplot as plt
import numpy as np

# === 假設三組 ablation 的平均時間 (秒) ===
io_times   = [1.14, 1.77, 1.66]   # Case1, Case2, Case3
comm_times = [5.46, 2.80, 2.73]
comp_times = [4.92, 4.43, 4.50]

labels = ["Case 1\n(no check, no BS)", 
          "Case 2\n(check only)", 
          "Case 3\n(check + BS)"]

x = np.arange(len(labels))
width = 0.5

# 顏色配置
colors = {"io":"skyblue", "comm":"salmon", "comp":"lightgreen"}

fig, ax = plt.subplots(figsize=(8,6))

# 畫堆疊長條
bars_io = ax.bar(x, io_times, width, label="I/O", color=colors["io"])
bars_comp = ax.bar(x, comp_times, width, bottom=io_times, label="Compute", color=colors["comp"])
bars_comm = ax.bar(x, comm_times, width, bottom=np.array(io_times)+np.array(comp_times),
                   label="Comm", color=colors["comm"])

# 計算總時間
totals = np.array(io_times) + np.array(comp_times) + np.array(comm_times)

# 在每個長條頂部加上數字標籤
for i, total in enumerate(totals):
    ax.text(x[i], total + 0.3, f"{total:.1f}", 
            ha="center", va="bottom", fontsize=11, fontweight="bold")

# 標籤設置
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
ax.set_ylabel("Runtime (seconds)", fontsize=13, fontweight="bold")
ax.set_title("Ablation Study: Time Profile", fontsize=15, fontweight="bold")

ax.legend(loc="upper right", ncol=3, fontsize=11, frameon=False)

plt.tight_layout()
plt.show()
