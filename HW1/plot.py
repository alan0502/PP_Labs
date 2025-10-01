import numpy as np
import matplotlib.pyplot as plt

# ===== Single-node runs (每個 case 三次) =====
cpu_runs_single = [
    [22.52, 22.40, 22.60],  # case 1
    [12.89, 13.01, 12.75],  # case 2
    [7.86, 7.90, 7.95],     # case 4
    [6.12, 6.20, 6.05]      # case 8
]
comm_runs_single = [
    [0.0, 0.0, 0.0],        
    [0.90, 0.85, 0.95],     
    [2.62, 2.70, 2.55],     
    [4.10, 4.05, 4.20]      
]
io_runs_single = [
    [2.54, 2.50, 2.60],     
    [2.67, 2.60, 2.70],     
    [2.04, 2.10, 2.00],     
    [1.37, 1.40, 1.35]      
]

# 平均
cpu_single = np.mean(cpu_runs_single, axis=1)
comm_single = np.mean(comm_runs_single, axis=1)
io_single   = np.mean(io_runs_single, axis=1)
labels_single = [1, 2, 4, 8]

# ===== Multi-node runs (舉例假資料) =====
cpu_runs_multi = [
    [7.89, 7.95, 7.88],   # case=4
    [5.78, 5.80, 5.75],   # case=8
    [5.93, 5.90, 6.00],   # case=16
    [4.42, 4.45, 4.40]    # case=32
]
comm_runs_multi = [
    [2.60, 2.65, 2.55],
    [3.15, 3.10, 3.20],
    [6.56, 6.50, 6.60],
    [6.40, 6.35, 6.45]
]
io_runs_multi = [
    [0.96, 0.95, 0.97],
    [0.88, 0.90, 0.87],
    [0.82, 0.83, 0.81],
    [0.78, 0.79, 0.77]
]

cpu_multi = np.mean(cpu_runs_multi, axis=1)
comm_multi = np.mean(comm_runs_multi, axis=1)
io_multi   = np.mean(io_runs_multi, axis=1)
labels_multi = [4, 8, 16, 32]

# ===== 畫圖 (一左一右) =====
fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=True)

colors = {'io': 'skyblue', 'cpu': 'lightgreen', 'comm': 'salmon'}

# --- Single-node ---
x1 = np.arange(len(labels_single))
axes[0].bar(x1, io_single, label="I/O", color=colors['io'])
axes[0].bar(x1, cpu_single, bottom=io_single, label="CPU", color=colors['cpu'])
axes[0].bar(x1, comm_single, bottom=io_single+cpu_single, label="Comm", color=colors['comm'])

axes[0].set_xticks(x1)
axes[0].set_xticklabels(labels_single, fontsize=12, fontweight="bold")
axes[0].set_xlabel("# of processes\n(1 node)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Runtime (seconds)", fontsize=14, fontweight="bold")
axes[0].set_title("Single Node (avg of runs)", fontsize=16, fontweight="bold")

# --- Multi-node ---
x2 = np.arange(len(labels_multi))
axes[1].bar(x2, io_multi, label="I/O", color=colors['io'])
axes[1].bar(x2, cpu_multi, bottom=io_multi, label="CPU", color=colors['cpu'])
axes[1].bar(x2, comm_multi, bottom=io_multi+cpu_multi, label="Comm", color=colors['comm'])

axes[1].set_xticks(x2)
axes[1].set_xticklabels(labels_multi, fontsize=12, fontweight="bold")
axes[1].set_xlabel("# of processes\n(4 nodes)", fontsize=14, fontweight="bold")
axes[1].set_title("Multi Node (avg of runs)", fontsize=16, fontweight="bold")

# 共用 legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=3, fontsize=12, frameon=False)

plt.tight_layout(rect=[0,0,1,0.93])
plt.suptitle("Time Profile: Single vs Multi Node", fontsize=18, fontweight="bold")
plt.show()
