import torch
import numpy as np
import matplotlib.pyplot as plt
from net import BallDirectionNet

# 类别映射
dir_map = {0: 'down', 1: 'up', 2: 'left', 3: 'right'}

# 加载模型
model = BallDirectionNet()
model.load_state_dict(torch.load("ball_direction_net1.pth", map_location="cpu"))
model.eval()

# 加载数据
all_data = np.load("all_encoded_data.npy")  # (4000, 36, 36)
num_samples = all_data.shape[0]

# 随机选取10个不同的样本索引
np.random.seed(42)  # 固定随机种子，保证可复现
random_indices = np.random.choice(num_samples, size=10, replace=False)

plt.figure(figsize=(15, 4))
for i, idx in enumerate(random_indices):
    sample = all_data[idx]  # (36, 36)
    sample_tensor = torch.tensor(sample, dtype=torch.float32).view(1, -1)  # (1, 1296)
    with torch.no_grad():
        output = model(sample_tensor)
        pred = torch.argmax(output, dim=1).item()
    print(f"第{idx}个样本预测类别为: {pred} ({dir_map[pred]})")
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample, cmap='gray')
    plt.title(f"{dir_map[pred]}")
    plt.axis('off')
plt.tight_layout()
plt.show()