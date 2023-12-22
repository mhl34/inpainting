import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.module):
    def __init__(self):
        super(encoder, self).__init__()
        # input: 224 x 224 x 3
        # output: 54 x 54 x 96
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4)
        # input: 54 x 54 x 96
        # output: 26 x 26 x 96
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        # input: 26 x 26 x 96
        # output: 26 x 26 x 256
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)
        # input: 26 x 26 x 256
        # output: 12 x 12 x 256
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        # input: 12 x 12 x 256
        # output: 12 x 12 x 384
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1)
        # input: 12 x 12 x 384
        # output: 12 x 12 x 384
        self.conv4 = nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1, padding = 1)
        # input: 12 x 12 x 384
        # output: 12 x 12 x 256
        self.conv5 = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        # input: 12 x 12 x 384
        # output: 6 x 6 x 256
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        # sequence for AlexNet Encoder
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = self.pool3(out)
        return out
