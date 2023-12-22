import torch
import torch.nn as nn
import torch.nn.functional as F

class decoder(nn.module):
    def __init__(self):
        super(decoder, self).__init__()
        # series of 5 upsamplings back to the original image size
        # 12 -> 26 -> 54 -> 224
        # input: 6 x 6 x 256
        # output: 12 x 12 x 256
        self.upconv1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)
        # input: 12 x 12 x 256
        # output: 12 x 12 x 384
        self.conv1 = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1)
        # input: 12 x 12 x 384
        # output: 26 x 26 x 256
        self.upconv2 = nn.ConvTranspose2d(in_channels = 384, out_channels = 256, kernel_size = 4, stride = 2)
        # input: 26 x 26 x 256
        # output: 54 x 54 x 96
        self.upconv3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 96, kernel_size = 4, stride = 2)
        # input: 54 x 54 x 96
        # output: 110 x 110 x 24
        self.upconv4 = nn.ConvTranspose2d(in_channels = 96, out_channels = 24, kernel_size = 4, stride = 2)
        # input: 110 x 110 x 24
        # output: 224 x 224 x 3
        self.upconv5 = nn.ConvTranspose2d(in_channels = 24, out_channels = 3, kernel_size = 6, stride = 2)

    def forward(self, x):
        out = self.upconv1(x)
        out = self.conv1(out)
        out = self.upconv2(out)
        out = self.upconv3(out)
        out = self.upconv4(out)
        out = self.upconv5(out)
        return out