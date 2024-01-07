#非线性激活，一般有relu和sigmoid
import torch
from torch import nn
from torch import ReLU

input=torch.tensor([[1,-0.5],
                    [-1,3]])
output=torch.reshape(input,(-1,1,2,2))
print(output.shape)

class YinKang(nn.Module):
    def __init__(self):
        super(YinKang,self).__init__()
        self.relu1=ReLU()

    def forward(self,input):
        output=self.relu1(input)
        return output
    
yinkang=YinKang()
output=yinkang(input)
print(output)