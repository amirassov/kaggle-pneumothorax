import torch
from torch import nn

EPSILON = 1e-6


class StableBCELoss(nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input = input.float().view(-1)
        target = target.float().view(-1)
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
