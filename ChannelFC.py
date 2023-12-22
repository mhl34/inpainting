import torch
import torch.nn as nn

class ChannelFC(nn.Module):
    def __init__(self, numFeatures, num_channels):
        super(ChannelFC, self).__init__()
        self.num_channels = num_channels
        self.numFeatures = numFeatures
        self.fc_layers = nn.ModuleList([
            nn.Linear(num_channels, num_channels) for _ in range(numFeatures)
        ])

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x_reshaped = x.view(batch_size, self.numFeatures, height, width)
        fc_outputs = [fc(x_reshaped[:,i,:,:].view(batch_size, -1)) for i , fc in enumerate(self.fc_layers)]
        fc_outputs = [i.view(batch_size, height, width) for i in fc_outputs]
        output = torch.stack(fc_outputs, dim = 1)
        return output
