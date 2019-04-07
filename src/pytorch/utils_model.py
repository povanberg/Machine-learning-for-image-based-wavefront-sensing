import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
    
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
