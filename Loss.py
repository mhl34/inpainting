import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, pred, target):
        l2loss = nn.MSELoss()
        return l2loss(pred, target)