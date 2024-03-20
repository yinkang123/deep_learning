import torchvision
from torch import nn
import sys
sys.path.append("../")
from configs import CIFAR10_path

#pretrained 参数决定了是否加载预训练的模型权重。
#如果 pretrained=True，那么将会加载在 ImageNet 数据集上预训练的模型权重。
#如果 pretrained=False，则会初始化一个新的模型，所有的模型权重都会被随机初始化。

# 在以前的版本中，你可能会这样加载一个预训练的模型：
# model = torchvision.models.vgg16(pretrained=True)
# 现在，你应该这样做：
# model = torchvision.models.vgg16(weights=torchvision.models.vgg16.WEIGHTS_CLASSIFICATION)
# 这里，WEIGHTS_CLASSIFICATION 是一个枚举值，表示你想要加载用于图像分类任务的预训练权重。对于其他模型或任务，你可能需要使用不同的枚举值。

vgg16_false = torchvision.models.vgg16(pretrained=False)
#设置为True会进行下载
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(CIFAR10_path, train=True, 
                                          transform=torchvision.transforms.ToTensor(), download=False)
print("********************************")
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
#想加到classifier里面
#vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)
print("********************************")
#修改
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print("********************************")
print(vgg16_false)


