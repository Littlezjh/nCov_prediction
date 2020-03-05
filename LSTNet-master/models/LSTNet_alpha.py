import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidR = args.hidRNN;
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;
        self.skip = args.skip;
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = args.highway_window
        self.alpha=args.alpha
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p = args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        # if(self.alpha==True):
        #     self.dd
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
 
    def forward(self, x):
        # print(x)
        batch_size = x.size(0);             #x:(13<batch_size,7,3)
        #CNN
        c = x.view(-1, 1, self.P, self.m);  #c:(13,1,7,3)
        c = F.relu(self.conv1(c));          #c:(13,100,5,1)  使用cnn卷积，将最后一维由m维变为1维

        c = self.dropout(c);
        c = torch.squeeze(c, 3);            #c:(13,100,5)   去除了最后一维
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();#r:(5,13,100)   变换维度，contiguous生成一个新的拷贝
        _, r = self.GRU1(r);                #r:(1,13,100)   GRU隐藏单元=CNN隐藏单元=100
        r = self.dropout(torch.squeeze(r,0));#r:(13,100)    去掉第一维



        #skip-rnn

        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            s = s.permute(2,0,3,1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        res = self.linear1(r);          #res:(13,3)
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];     #z:(13,7,3) 当window大于highway_window时，会缩减
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);  #z:(39,7)
            z = self.highway(z);        #z:(39,1)
            z = z.view(-1,self.m);      #z:(13,3)
            res = res + z;

        if(self.alpha==True):
            alpha=torch.zeros(batch_size,self.m)
            que=self.m-1
            for j in range(batch_size):
                a=x[j,:,que]
                me=[]
                for i in range(self.P-2):
                    b = (torch.log(a[i + 2,] / a[i + 1]) / torch.log(a[ i + 1] / a[ i])).item()
                    me.append(b)
                median=np.median(me)
                quezhen=(a[self.P-1] / a[self.P-2]).pow(median) * a[self.P-1]
                alpha[j][que]=quezhen
            # print(x[:,:,2])
            # print(alpha)
            res=res+alpha

        if (self.output):
            res = self.output(res);
        return res;
    
        
        
        
