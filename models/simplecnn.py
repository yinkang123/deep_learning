#这个model是在CIFAR10_train.py里面使用的
import torch
from torch import nn

# 搭建神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    my_model = SimpleCNN()
    input = torch.ones((64, 3, 32, 32))
    output = my_model(input) #output的shape是[64, 10]，其中64是batch_size, 10是类别数
    # 64是batch_size, 10是类别数
    print(output.shape)