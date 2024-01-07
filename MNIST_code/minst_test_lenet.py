import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from lenet import LeNet5

model=LeNet5()
# 加载训练好的模型
model.load_state_dict(torch.load('../model_weights/lenet_epoch5.pth',map_location=torch.device('cpu')))
model.eval()  # 将模型设置为评估模式

# 预处理输入图像
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 如果图像是彩色，转为灰度图
    transforms.Resize((32,32)),  # 将图像调整为模型输入的大小
    transforms.ToTensor(),  # 将图像转换为张量
])

# 导入测试图像
image_path = '../dataset/temp/7.png'
input_image = Image.open(image_path).convert('L')  # 'L'表示将图像转为灰度图
input_tensor = transform(input_image).unsqueeze(0)  # 添加一个维度表示 batch_size

# 运行模型进行预测
with torch.no_grad():
    output = model(input_tensor)

# 获取预测结果
_, predicted_class = torch.max(output, 1)
predicted_class = predicted_class.item()

# 显示图像和预测结果
plt.imshow(input_image, cmap='gray')
plt.title(f'Predicted Class: {predicted_class}')
plt.show()
