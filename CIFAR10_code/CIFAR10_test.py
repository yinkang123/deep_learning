import torch
import torchvision
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt

class YinKang(nn.Module):
    def __init__(self):
        super(YinKang, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("../model_weights/yinkang_29_gpu.pth", map_location=torch.device('cpu'))

model.eval()

# 导入测试图像
image_path = '../dataset/temp/dog.png'
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