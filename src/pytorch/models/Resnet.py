import torch
import numpy as np
import aotools
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet = models.resnet50(pretrained=True)   
        
        for param in self.resnet.parameters():
            param.requires_grad = True
       
        # Input size 2x128x128 -> 2x224x224
        first_conv_layer = [nn.Conv2d(2, 3, kernel_size=1, stride=1, bias=True),
                            nn.AdaptiveMaxPool2d(224),
                            self.resnet.conv1]
        self.resnet.conv1= nn.Sequential(*first_conv_layer)

        # Fit classifier
        self.resnet.fc = nn.Sequential(
                                nn.Linear(2048, 100),
                                #nn.ReLU(inplace=True),
                                #nn.BatchNorm1d(1024),
                                #nn.Linear(1024, 1024),
                                #nn.ReLU(inplace=True),
                                #nn.BatchNorm1d(1024),
                                #nn.Linear(1024, 20)
                            )    
    
        self.phase2dlayer = Phase2DLayer(100,128)

    def forward(self, x):
        # 128x128x2
        z = self.resnet(x)
        phase = self.phase2dlayer(z)
        return phase

class Phase2D(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, z_basis):
        ctx.z_basis = z_basis.cuda()
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
