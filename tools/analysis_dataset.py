


# analysis dataset defect class and distru

import os
from pathlib import Path
import numpy as np
import matplotlib as plt
from matplotlib.ticker import PercentFormatter

root = r'\\10.211.64.54\dataset\SBU_AOI'
label_ok = ['滚轮印', '毛刷印', '边缘未洗到']
folder = [i for i in os.listdir(root) if 'gz' not in i]
total_class_count = {}
total_class_type = []

# label 处理
for folder_i in folder:
    fold_path = Path(root) / folder_i
    class_type = os.listdir(fold_path)
    total_class_type = total_class_type + class_type

# 取并集
total_class_type = set(total_class_type) 


# 数量统计
for folder_i in folder:
    fold_path = Path(root) / folder_i
    class_type = os.listdir(fold_path)
    class_count = {class_i: len(os.listdir(Path(fold_path) / class_i)) if class_i in class_type else 0 for class_i in total_class_type}
    total_class_count[folder_i] = class_count 

x_labels = [f'OK-{i}' if i in label_ok else f'NG-{i}' for i in total_class_type]
x = range(len(folder))
bottom_y = np.zeros(len(folder))
sums = [sum(i.values()) for i in total_class_count.values()]
index = 0
data = []

# data 转array 
for key_i in total_class_count:
    y = [round(a / sums[index], 2) for a in total_class_count[key_i].values()]
    data.append(y)
    index += 1 
data_arr = np.array(data,dtype=np.float32)


# 绘图
plt.subplots(figsize=((8,6)))
plt.rcParams['font.sans-serif'] = ['SimHei']

for i in range(len(data[0])):
    y = data_arr[:,i]
    plt.bar(x, y, 0.35, bottom=bottom_y, label=x_labels[i])
    for xi, yl, yi in zip(x, bottom_y, y):
        if yi >= 0.01:
            plt.text(xi, yl + yi / 3, f'{yi*100:.1f}%', ha='center', va='center')
    bottom_y = y + bottom_y
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))

plt.legend(bbox_to_anchor=(1, 0.8))
plt.xticks(x, total_class_count.keys())
plt.subplots_adjust(right=0.8)
plt.show()






