
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from utils.random_image import random_image_from_dataset
from models.vgg16 import vgg16

# test_pth=r'D:/aaa_MyGit/deep_learning/dataset/test/dog.png'#设置可以检测的图像
# test=Image.open(test_pth).convert('RGB')

test= random_image_from_dataset('D:\Datasets\mini_cats_and_dogs')#随机选择一张图片
test=test.convert('RGB')#转换为RGB格式

'''处理图片'''
transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
image=transform(test)
'''加载网络'''
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")#CPU与GPU的选择
net =vgg16()#输入网络
model=torch.load(r"D:/aaa_MyGit/deep_learning/CatsVSDogs_code/checkpoints/DogandCat8.pth",map_location=device)#已训练完成的结果权重输入
net.load_state_dict(model)#模型导入
net.eval()#设置为推测模式

image=torch.reshape(image,(1,3,224,224))#四维图形，RGB三个通
with torch.no_grad():
    out=net(image)
out=F.softmax(out,dim=1)#softmax 函数确定范围
out=out.data.cpu().numpy()
print(out)
a=int(out.argmax(1))#输出最大值位置
plt.figure()
list=['Cat','Dog']
plt.suptitle("Classes:{}:{:.1%}".format(list[a],out[0,a]))#输出最大概率的道路类型
plt.imshow(test)
plt.show()