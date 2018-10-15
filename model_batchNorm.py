import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.FloatTensor')

# VGG like
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
       
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv11_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv22_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv33 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv33_bn = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(128 * 16 * 16, 20)
 
    def forward(self, x):
        # 128x128x2
        x = self.conv1_bn(self.conv1(x))    
        x = F.relu(x) 
        x = self.pool(self.conv11_bn(self.conv11(x)))       
        x = F.relu(x) 
        # 64x64x64
        x = self.conv2_bn(self.conv2(x))    
        x = F.relu(x) 
        x = self.pool(self.conv22_bn(self.conv22(x)))       
        x = F.relu(x)                  
        # 32x32x128          
        x = self.conv3_bn(self.conv3(x))    
        x = F.relu(x) 
        x = self.pool(self.conv33_bn(self.conv33(x)))       
        x = F.relu(x)    
        # 16x16x128 
        x = x.view(-1, 128 * 16 * 16)  
        # 1x(16x16x128)
        x = self.fc1(x)                 
        return x