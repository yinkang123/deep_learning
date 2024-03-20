import torch
# 方式1-》保存方式1，加载模型
#这种方法有时候报错，需要加载模型的类
import torchvision
from torch import nn
# import sys
# sys.path.append("../")

model = torch.load("../weights/vgg16_method1.pth")
print(model)
print('***********************')
# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("../weights/vgg16_method2.pth"))
print(vgg16)

