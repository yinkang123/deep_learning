import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 *7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))])
                             
data_train = MNIST('../dataset/MNIST',
                   download=True,
                   transform=transform)
    
data_test = MNIST('../dataset/MNIST',
                  train=False,
                  download=True,
                  transform=transform)
data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=8,drop_last=True)
data_test_loader = DataLoader(data_test, batch_size=64, num_workers=8,drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Determine the device to use
net = SimpleCNN().to(device)  # Move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

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
    torch.save(net.state_dict(), f"SimpleCNN_epoch{epoch}.pth")

def main():
    for e in range(1, 6):
        train_and_test(e)

if __name__ == '__main__':
    main()