import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.FloatTensor')

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)    
    

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
       
        self.Conv2d_1a_3x3 = BasicConv2d(2, 16, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(16, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_2c_3x3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        
        self.Mixed_A = InceptionA(64, pool_features=4)
        
        self.Conv2d_3a_3x3 = BasicConv2d(228, 124, kernel_size=3, padding=2)
        self.Conv2d_3b_3x3 = BasicConv2d(124, 64, kernel_size=3, padding=2)
        
        self.Mixed_B = InceptionB(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc = nn.Sequential(
                        nn.Linear(64*8*8, 124),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5),
                        #nn.Linear(1024, 256),
                        #nn.ReLU(inplace=True),
                        #nn.Dropout(p=0.5),
                        nn.Linear(124, 20)
                    )
        
    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)  
        x = self.Conv2d_2c_3x3(x)   
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_A(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(-1, 64 * 8 * 8) 
        x = self.fc(x)
        return x
    
