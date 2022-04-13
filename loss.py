import torch
import torch.nn as nn

class loss_s(nn.Module):
    def __init__(self):
        super(loss_s, self).__init__()

    def forward(self, s, t, C, loss_w):
        n = len(s) - 1
        density = 0.001 # hyper-parameter
        beta = 1 - t*0.05
        u_s = density*beta*(1 - C)*loss_w

        loss = loss_w/n + u_s
        loss = torch.mean(loss)

        return loss