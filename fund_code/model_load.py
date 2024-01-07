import torch
from model_save import *
# 方式1-》保存方式1，加载模型
import torchvision
from torch import nn

model = torch.load("../model_weights/vgg16_method1.pth")
print(model)
print('***********************')
# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("../model_weights/vgg16_method2.pth"))
print(vgg16)

