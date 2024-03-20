import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sys
sys.path.append("../")
from configs import MNIST_path
from models.lenet import LeNet5

data_train = MNIST(MNIST_path,
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST(MNIST_path,
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Determine the device to use
net = LeNet5().to(device)  # Move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

def train(epoch):
    net.train()
    for batch_idx, (images, labels) in enumerate(data_train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        output = net(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))

def train_and_test(epoch):
    train(epoch)
    test()
    torch.save(net.state_dict(), f"../weights/MNIST_with_lenet{epoch}.pth")

def main():
    for e in range(1, 6):
        train_and_test(e)

if __name__ == '__main__':
    main()