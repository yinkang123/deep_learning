import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数
#保存的是整个模型，包括模型的结构和参数。这意味着你可以直接加载整个模型，而无需预先知道模型的结构
torch.save(vgg16, "../weights/vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐）
#这种方法只保存模型的参数，不保存模型的结构。这意味着在加载模型时，你需要先创建一个与保存参数时相同结构的模型，然后加载参数
#pretrained=false,所以不会下载预训练的模型权重,所以这里保存的是随机数
torch.save(vgg16.state_dict(), "../weights/vgg16_method2.pth")
