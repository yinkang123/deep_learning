#maxpool目的：保留数据特征，降维，减少数据量
import torch
from torch import nn
from torch.nn import MaxPool2d

input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]],dtype=torch.float32)
input=torch.reshape(input,(-1,1,5,5))
print(input.shape)

class YinKang(nn.Module):
    def __init__(self):
        super(YinKang,self).__init__()
        #池化默认的步长是kernel_size
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool1(input=input)
        return output
    
yinkang=YinKang()
output=yinkang(input)
print(output)