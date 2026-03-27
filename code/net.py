import torch
import torch.nn as nn
import torch.nn.functional as F
#这段代码是建立的神经网络

class BallDirectionNet(nn.Module):
    def __init__(self):
        super(BallDirectionNet, self).__init__()
        self.fc1 = nn.Linear(36*36, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 示例：
# model = BallDirectionNet()
# out = model(torch.randn(8, 36, 36))  # batch=8
# print(out.shape)  # torch.Size([8, 4])
