import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hw = args.highway_window
        self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
    def forward(self, x):
        batch_size = x.size(0);

        z = x[:, -self.hw:, :];
        z = z.permute(0,2,1).contiguous().view(-1, self.hw);
        z = self.highway(z);
        z = z.view(-1,self.m);
        res=z
        if (self.output):
            res = self.output(z);
        return res;
    
        
        
        
