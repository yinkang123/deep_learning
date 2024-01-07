import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的通道数，因为minist数据集是灰度图像只有一个通道
                out_channels=16,  # 输出通道数，我们可以自己定义，这里设定为16，也就是说有16个卷积核，提取16种特征
                kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        return output

model=CNN()
# 加载训练好的模型
model.load_state_dict(torch.load('../model_weights/cnn2_gpu.pth',map_location=torch.device('cpu')))
model.eval()  # 将模型设置为评估模式

# 预处理输入图像
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 如果图像是彩色，转为灰度图
    transforms.Resize((28, 28)),  # 将图像调整为模型输入的大小
    transforms.ToTensor(),  # 将图像转换为张量
])

# 导入测试图像
image_path = '../dataset/temp/8.png'
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
