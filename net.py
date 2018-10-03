import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.DoubleTensor')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(2, 20, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(40, 20, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(16 * 16 * 20, 256)
        self.fc2 = torch.nn.Linear(256, 20)

    def forward(self, x):
        # 128x128x2
        x = F.relu(self.conv1(x))         # 128x128x20
        x = self.pool(x)                  # 64x64x20    Todo: bottleneck
        x = F.relu(self.conv2(x))         # 64x64x40
        x = self.pool(x)                  # 32x32x40
        x = F.relu(self.conv3(x))         # 32x32x20
        x = self.pool(x)                  # 16x16x20
        x = x.view(-1, 20 * 16 * 16)      # 1x5120      (Reshape for fully connected layer)
        x = F.relu(self.fc1(x))           # 1x256
        x = self.fc2(x)                   # 1x20        (Zernike coeffs)
        return x