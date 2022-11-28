import torch
import torch.nn as nn

class network(nn.Module):
    def __init__(self, inputdim1, inputdim2, inputdim3):
        super(network, self).__init__()
        self.linear_cat = nn.Linear(inputdim1 + inputdim2 + inputdim3, 32)
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(32, 1)

    def forward(self, src1, src2, src3):
        src = torch.cat((src1, src2, src3), dim=1)
        out_pool = self.linear_cat(self.drop(src))
        out =  torch.sigmoid(self.linear(self.drop(out_pool)))
        return out.squeeze(1)
    



