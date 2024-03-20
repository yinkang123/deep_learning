import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
import sys
sys.path.append("../")
from configs import CIFAR10_path

dataset = torchvision.datasets.CIFAR10(CIFAR10_path, train=False, 
                                       transform=torchvision.transforms.ToTensor(),download=False)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class YinKang(nn.Module):
    def __init__(self):
        super(YinKang, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = torch.flatten(input)  # Flatten the input tensor
        output = self.linear1(output)
        return output

yinkang = YinKang()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape) # torch.Size([64, 3, 32, 32])
    output = yinkang(imgs)  # Pass the input tensor directly to the model
    print(output.shape)     # torch.Size([10])