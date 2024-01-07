import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
#设置为True会进行下载
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('../dataset/CIFAR10', train=True, 
                                          transform=torchvision.transforms.ToTensor(), download=True)

vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
#想加到classifier里面
#vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

#修改
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)


