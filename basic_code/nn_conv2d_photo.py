import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("../")
from configs import CIFAR10_path

dataset = torchvision.datasets.CIFAR10(CIFAR10_path, train=False, 
                                       transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset, batch_size=64)

class YinKang(nn.Module):
    def __init__(self):
        super(YinKang, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

yinkang = YinKang()

writer = SummaryWriter("../logs/conv2d")

step = 0
for data in dataloader:
    imgs, targets = data
    output = yinkang(imgs)
    print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    print(output.shape) # torch.Size([64, 6, 30, 30])
    writer.add_images("input", imgs, step)
    #32-3+1=30
    #老师报错所以reshape为三通道
    output = torch.reshape(output, (-1, 3, 30, 30))  # torch.Size([64, 6, 30, 30])  -> [xxx, 3, 30, 30]
    writer.add_images("output", output, step)

    step = step + 1


