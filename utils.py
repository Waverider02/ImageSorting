import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

# 将多个基本参数拼接成字符串
def to_msg(*args):
    total_msg = ""
    for arg in args:
        total_msg += str(arg)
        total_msg += " "
    return total_msg

# 将0-9的数字编码成1*10的数组
def one_hot(label,depth=10):
    out = torch.zeros(label.size(0),depth)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out

# 绘制图像
def plot_imag(img,label,name):
    plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307,cmap='gray',interpolation='none')
        plt.title("{}: {}".format(name,label[i].item()))
        plt.xticks([])
        plt.yticks([])

# 获取数据
def get_data(batch_size=128): # 使用minist数据集
    train_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("mnist_data",train=True,download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,),(0.3081,)
                                    )
                                ])),
        batch_size=batch_size,shuffle=True # shuffle=True每个epoch打乱训练集
    )
    test_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("mnist_data/",train=False,download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,),(0.3081,)
                                    )
                                ])),
        batch_size=batch_size,shuffle=True
    )
    return (train_data,test_data)

"""全连接神经网络"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,64) # 两层隐藏层, 大小分别为256,64
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):
        # [b,1,28,28]
        x = x.view(x.size(0),28*28)
        # [b,28*28]
        x = F.relu(self.fc1(x))
        # [b,256]
        x = F.relu(self.fc2(x))
        # [b,64]
        x = self.fc3(x)
        # [b,10]
        return x

"""CNN神经网络"""
# nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
# in_channels:输入的四维张量[N, C, H, W]中的C，即输入张量的channels数。
# C 对于图像而言可以理解为颜色通道。
# out_channels:输出的四维张量的channels数。
# kernel_size:卷积核大小。
# stride:卷积核每次平移的步长, 默认为1。
# padding:图像边框填充, 默认为0。
# 输出图像尺寸:
# out = (in+2*padding-kernel_size)/stride+1

# max_pool2d(input,kernel_size,stride,padding)
# 输出图像尺寸:
# out = (in+2*padding-kernel_size)/stride+1

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,4,3,3,1)
        self.conv2 = nn.Conv2d(4,12,3,1,1)
        self.fc1 = nn.Linear(12*10*10,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        # [b,1,28,28]
        x = torch.relu(self.conv1(x))
        # [b,4,10,10]
        x = torch.max_pool2d(x,3,1,1)
        # [b,4,10,10]
        x = torch.relu(self.conv2(x))
        # [b,12,10,10]
        x = torch.max_pool2d(x,3,1,1)
        # [b,12,10,10]
        x = x.view(-1,12*10*10)
        # [b,12*10*10]
        x = torch.relu(self.fc1(x))
        # [b,128]
        x = self.fc2(x)
        # [b,10]
        return x
