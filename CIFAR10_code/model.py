#这个model是在CIFAR10_train.py里面使用的
import torch
from torch import nn

# 搭建神经网络
class YinKang(nn.Module):
    def __init__(self):
        super(YinKang, self).__init__()
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
    yinkang = YinKang()
    input = torch.ones((64, 3, 32, 32))
    output = yinkang(input)
    # 64是batch_size, 10是类别数
    print(output.shape)