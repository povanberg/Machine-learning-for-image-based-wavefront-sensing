import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv_a1 = BasicConv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv_a2 = BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1)
        
        self.conv_b1 = BasicConv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_b2 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv_c1 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_c2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv_d1 = BasicConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.conv_e1 = BasicConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        self.conv_f1 = BasicConv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(4*4*256, 256)
        self.fc2 = torch.nn.Linear(256, 20)

    def forward(self, x):
        # 128x128x2
        x = self.conv_a1(x) # 128x128x16
        x = self.conv_a2(x)
        x = self.pool(x)
        x = self.conv_b1(x) # 64x64x32
        x = self.conv_b2(x)
        x = self.pool(x)
        x = self.conv_c1(x) # 32x32x64
        x = self.conv_c2(x)
        x = self.pool(x)
        x = self.conv_d1(x) # 16x16x128
        x = self.conv_d2(x)
        x = self.pool(x)
        x = self.conv_e1(x) # 8x8x256
        x = self.pool(x)
        x = x.view(-1, 4*4*256) 
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)    