import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class YinKang(nn.Module):
    def __init__(self):
        super(YinKang, self).__init__()
        #如果不用sequential，需要在这里定义各种操作，然后在forward里面使用各种操作
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),#padding是为了让H X W=32X32,根据公式计算得到
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

yinkang = YinKang()
print(yinkang)
input = torch.ones((64, 3, 32, 32))
output = yinkang(input)
print(output.shape)

writer = SummaryWriter("../logs/seq")
writer.add_graph(yinkang, input)
writer.close()
