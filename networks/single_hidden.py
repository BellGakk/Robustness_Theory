import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys

def param_init(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        init.normal_(m.weight, mean=0.0, std=1.0)
        init.normal_(m.bias, mean=0.0, std=1.0)

    return classname

class MLP(nn.Module):
    def __init__(self, in_plane, inter_plane, out_class):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(in_plane, inter_plane)
        self.mlp2 = nn.Linear(inter_plane, out_class)
        self.norm = nn.BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax()
    def forward(self, x):
        out = self.softmax(self.mlp2(F.relu(self.norm(self.mlp1(x)))))
        
        return out

if __name__ == '__main__':
    mlp = MLP(1024, 10000, 10)
    k = 0.5
    for name, module in mlp._modules.items():
        print(name)
        if name == 'mlp1':
            init.normal_(module.weight, mean=0.0, std=k*1.0)
            init.normal_(module.bias, mean=0.0, std=k*1.0)
        elif name == 'mlp2':
            init.normal_(module.weight, mean=0.0, std=k*1.0)
            init.normal_(module.bias, mean=0.0, std=k*1.0)
    
    

    

    