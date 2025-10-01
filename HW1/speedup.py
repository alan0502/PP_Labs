import matplotlib.pyplot as plt
import numpy as np

# single node 下的 speedup 數據
#labels = [1, 2, 4, 8]   # process 數
#speedup = [1, 1.52, 1.99, 2.15]

# multi node 下的 speedup 數據 
# process 數 (用來標籤) 
labels = [1, 4, 8, 16] 
# speedup 數據 
speedup = [1, 1.998, 2.18, 1.88]

# 用 index 當作等距的 x 軸
x = np.arange(len(labels))

plt.figure(figsize=(7,5))
plt.plot(x, speedup, marker='s', color='royalblue', linewidth=2, markersize=8, label='Measured')

# 理想線 (y = x)
plt.plot(x, labels, color='red', linestyle='--', linewidth=2, label='Ideal (y = x)')

# 等距的 xticks
plt.xticks(x, labels, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# 標籤與標題
plt.xlabel('# of processes\n(4 cores per node)', fontsize=14, fontweight='bold')
#plt.xlabel('# of processes\nsingle node', fontsize=14, fontweight='bold')
plt.ylabel('Speedup', fontsize=14, fontweight='bold')
plt.title('Strong Scalability: Speedup vs Processes', fontsize=16, fontweight='bold')

# 網格線 + 圖例
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()
