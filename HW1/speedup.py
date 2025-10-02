import matplotlib.pyplot as plt
import numpy as np

# --- Single node ---
labels_single = [1, 2, 4, 8]
basic_single = [1, 1.52, 1.99, 2.15]   # 原始 basic speedup
opt_single   = [1, 1.63, 2.14, 2.52]   # optimize speedup
ideal_single = labels_single           # 理想 speedup (y=x)

x1 = np.arange(len(labels_single))

# --- Multi node ---
labels_multi = [1, 4, 8, 16]
basic_multi = [1, 1.998, 2.18, 1.88]
opt_multi   = [1, 2.14, 2.73, 2.93]
ideal_multi = labels_multi

x2 = np.arange(len(labels_multi))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 不 sharey

# 左：Single node
axes[0].plot(x1, basic_single, marker='o', color='steelblue', linewidth=2, label='Basic')
axes[0].plot(x1, opt_single, marker='s', color='seagreen', linewidth=2, label='Optimized')
axes[0].plot(x1, ideal_single, linestyle='--', color='red', linewidth=2, label='Ideal (y=x)')
axes[0].set_xticks(x1)
axes[0].set_xticklabels(labels_single, fontsize=12, fontweight="bold")
axes[0].set_xlabel("# of processes\n(Single Node)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Speedup", fontsize=14, fontweight="bold")
axes[0].set_title("Single Node", fontsize=16, fontweight="bold")
axes[0].set_ylim(0, 8.5)   # 👈 y 軸範圍
axes[0].grid(True, linestyle='--', alpha=0.6)

# 右：Multi node
axes[1].plot(x2, basic_multi, marker='o', color='steelblue', linewidth=2, label='Basic')
axes[1].plot(x2, opt_multi, marker='s', color='seagreen', linewidth=2, label='Optimized')
axes[1].plot(x2, ideal_multi, linestyle='--', color='red', linewidth=2, label='Ideal (y=x)')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(labels_multi, fontsize=12, fontweight="bold")
axes[1].set_xlabel("# of processes\n(Multi Node)", fontsize=14, fontweight="bold")
axes[1].set_title("Multi Node", fontsize=16, fontweight="bold")
axes[1].set_ylim(0, 16.5)  # 👈 y 軸範圍
axes[1].grid(True, linestyle='--', alpha=0.6)

# 共用 legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12, frameon=False,
           bbox_to_anchor=(0.5, 0.95))

plt.suptitle("Strong Scalability: Basic vs Optimized", fontsize=18, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.92])
plt.show()