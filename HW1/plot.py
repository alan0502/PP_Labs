import matplotlib.pyplot as plt
import numpy as np

# ===== Single node =====
labels_single = [1, 2, 4, 8]
cpu_single = [22.52, 12.89, 7.86, 6.12]
comm_single = [0, 0.9, 2.62, 4.10]
io_single   = [2.54, 2.67, 2.04, 1.37]

#labels_single = [1, 4, 8, 12]
#cpu_single = [113.128, 43.1, 31.6, 28.2826]
#comm_single = [0, 14.039, 16.626, 18.363]
#io_single   = [2.482, 1.428, 1.435, 1.173]


# ===== Multi node =====
labels_multi = [4, 8, 16, 32]
cpu_multi = [7.89, 5.78, 5.93, 4.42]
comm_multi = [2.60, 3.15, 6.56, 6.40]
io_multi   = [0.96, 0.88, 0.82, 0.78]

fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=True)

# 顏色統一
colors = {'io': 'skyblue', 'cpu': 'lightgreen', 'comm': 'salmon'}

# --- 左邊 Single node ---
x1 = np.arange(len(labels_single))
axes[0].bar(x1, io_single, label='I/O', color=colors['io'])
axes[0].bar(x1, cpu_single, bottom=io_single, label='CPU', color=colors['cpu'])
axes[0].bar(x1, comm_single, bottom=np.array(io_single)+np.array(cpu_single), 
            label='Comm', color=colors['comm'])

axes[0].set_xticks(x1)
axes[0].set_xticklabels(labels_single, fontsize=12, fontweight='bold')
axes[0].set_xlabel('# of processes\n(1 node)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Runtime (seconds)', fontsize=14, fontweight='bold')
axes[0].set_title('Single Node', fontsize=16, fontweight='bold')

# --- 右邊 Multi node ---
x2 = np.arange(len(labels_multi))
axes[1].bar(x2, io_multi, label='I/O', color=colors['io'])
axes[1].bar(x2, cpu_multi, bottom=io_multi, label='CPU', color=colors['cpu'])
axes[1].bar(x2, comm_multi, bottom=np.array(io_multi)+np.array(cpu_multi), 
            label='Comm', color=colors['comm'])

axes[1].set_xticks(x2)
axes[1].set_xticklabels(labels_multi, fontsize=12, fontweight='bold')
axes[1].set_xlabel('# of processes\n(4 nodes)', fontsize=14, fontweight='bold')
axes[1].set_title('Multi Node', fontsize=16, fontweight='bold')

# y 軸刻度字體
for ax in axes:
    ax.tick_params(axis='y', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

# --- Legend 共用 ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=3, fontsize=12, frameon=False)

plt.tight_layout(rect=[0,0,1,0.93])
plt.suptitle('Time Profile: Single vs Multi Node', fontsize=18, fontweight='bold')
plt.show()
