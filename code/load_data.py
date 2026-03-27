import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
# 这段代码用于构建PyTorch的DataLoader，方便后续训练和测试模型
def get_ball_direction_dataloader(
    npy_path="all_encoded_data.npy",
    batch_size=64,
    test_ratio=0.2,
    random_seed=42
):
    # 读取数据
    data = np.load(npy_path)  # shape: (4000, 36, 36)

    # 生成标签
    labels = np.array([0]*1000 + [1]*1000 + [2]*1000 + [3]*1000)  # 0:down, 1:up, 2:left, 3:right

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_ratio, random_state=random_seed, stratify=labels
    )

    # 转换为Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 构建数据集和DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

