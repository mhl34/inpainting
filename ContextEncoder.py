from Encoder import Encoder
from Decoder import Decoder
from ChannelFC import ChannelFC
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextEncoder(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.channel_fc = ChannelFC(256, 36)
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.channel_fc(out)
        out = self.decoder(out)
        return out