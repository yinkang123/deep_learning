import torch
import torchvision
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from models.simplecnn import SimpleCNN
# 导入模型
model =SimpleCNN()
model.load_state_dict(torch.load("../weights/CIFAR10_with_SimpleCNN_29.pth",map_location=torch.device('cpu')))

model.eval()

# 导入测试图像
image_path = '../dataset/test/dog.png'
#搞半天出错，因为导入的图像通道数是4，包含透明度通道，所以要转换一下
image = Image.open(image_path).convert('RGB')  # 转换为 RGB 格式

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image_tensor = transform(image)
image_tensor = torch.unsqueeze(image_tensor, 0)  # 添加一个维度表示批次大小
# 运行模型进行预测
with torch.no_grad():
    output = model(image_tensor)

# 获取预测结果
_, predicted_class = torch.max(output, 1)
predicted_class = predicted_class.item()

# 显示图像和预测结果
plt.imshow(image)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.title(f'Predicted Class: {classes[predicted_class]}')
plt.show()