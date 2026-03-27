import numpy as np
import matplotlib.pyplot as plt

print("hello world")
# 读取npy文件
data = np.load("all_encoded_data.npy")  # 如果文件不在当前目录，请写绝对路径

print("数据形状:", data.shape)  # 应为 (4000, 36, 36)

# 随机挑选5个索引
indices = np.random.choice(data.shape[0], 5, replace=False)
print("随机选中的索引:", indices)

# 绘制5个图像
for i, idx in enumerate(indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(data[idx], cmap='gray')
    plt.title(f"Index {idx}")
    plt.axis('off')

plt.tight_layout()
plt.show()
