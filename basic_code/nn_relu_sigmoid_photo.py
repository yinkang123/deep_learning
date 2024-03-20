import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("../")
from configs import CIFAR10_path


dataset = torchvision.datasets.CIFAR10(CIFAR10_path, train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class YinKang(nn.Module):
    def __init__(self):
        super(YinKang, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

class KangYin(nn.Module):
    def __init__(self):
        super(KangYin, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.relu1(input)
        return output
    
yinkang = YinKang()
kangyin = KangYin()
writer = SummaryWriter("../logs/relu_and_sigmoid")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output_sigmoid = yinkang(imgs)
    writer.add_images("output_sigmoid", output_sigmoid, step)
    output_relu = yinkang(imgs)
    writer.add_images("output_relu", output_relu, step)
    step += 1

writer.close()


