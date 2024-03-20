import torchvision
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("../")
from configs import CIFAR10_path


dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
#可以将transform运用到dataset中每一个图片上
test_set = torchvision.datasets.CIFAR10(root=CIFAR10_path, train=False, transform=dataset_transform, download=False)

print(test_set[0])
#图像有十种分类
print(test_set.classes)
#有两个属性，img,target,直接赋值
img, target = test_set[0]
print(img.shape)
print(target)
#通过target索引到对应的类别
print(test_set.classes[target])



writer = SummaryWriter("../logs/basic_image")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()