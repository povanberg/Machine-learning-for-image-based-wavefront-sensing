import torch
import numpy as np
import aotools
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv_a1 = BasicConv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv_a2 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv_b1 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_b2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv_c1 = BasicConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_c2 = BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.conv_d1 = BasicConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.conv_e1 = BasicConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv_e2 = BasicConv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(4*4*512, 1024)
        #self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        #self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc3 = torch.nn.Linear(1024, 20)
       
        self.phase2dlayer = Phase2DLayer(20,128)

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
        x = self.conv_e1(x) # 8x8x512
        x = self.conv_e2(x) # 8x8x512
        x = self.pool(x)
        x = x.view(-1, 4*4*512) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        z_coeffs = self.fc3(x)
        phase = self.phase2dlayer(z_coeffs)
        return phase, z_coeffs

    
class Phase2D(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, z_basis):
        ctx.z_basis = z_basis.cpu() #.cuda()
        output = input[:,:, None, None] * ctx.z_basis[None, 1:,:,:]
        return torch.sum(output, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        dL_dy = grad_output.unsqueeze(1)
        dy_dz = ctx.z_basis[1:,:,:].unsqueeze(0)
        grad_input = torch.sum(dL_dy * dy_dz, dim=(2,3))
        return grad_input, None
    
class Phase2DLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(Phase2DLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.z_basis = aotools.zernikeArray(input_features+1, output_features, norm='rms')
        self.z_basis = torch.as_tensor(self.z_basis, dtype=torch.float32)
        
    def forward(self, input):
        return Phase2D.apply(input, self.z_basis)   
    
    
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)