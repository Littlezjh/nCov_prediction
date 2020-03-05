import argparse
import math
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from models import LSTNet
from models import ARmodel
from models import LSTNet_alpha
import numpy as np;
import importlib

from metrics import *;
from utils import *;
import Optim
from numpy import set_printoptions

# set_printoptions(threshold='nan')

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;
    
    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X);
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output));
            test = torch.cat((test, Y));
        
        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m);
    # rse = math.sqrt(total_loss / n_samples)/data.rse
    rse = total_loss / n_samples
    rae = (total_loss_l1/n_samples)/data.rae
    
    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    return rse, rae, correlation;

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train();
    total_loss = 0;
    n_samples = 0;
    # print(X.shape)
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad();
        output = model(X);
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale); #当前用的是MSEloss，
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.item();  #如果你有只有一个元素的张量，使用.item()来得到Python数据类型的数值
        n_samples += (output.size(0) * data.m);
    return total_loss / n_samples
    
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str,default='1',
                    help='location of the data file')
parser.add_argument('--model', type=str, default='LSTNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=100,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=7,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=6,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=3,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=150,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--skip', type=int, default=7)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
parser.add_argument('--alpha', type=bool, default=False)
args = parser.parse_args()

args.epochs=200
# args.data='allsum_data/data_hubei_2.26.txt'
# args.data='allsum_data/quezhen_nohubei_2.26.txt'
args.data='county_data/ezhou.txt'
args.save='save3/ezhou_ARmodel.pt'
# args.model='LSTNet_alpha'
# args.model='LSTNet'
args.model='ARmodel'

args.skip=-7
args.output_fun='Linear'
args.normalize=0
args.horizon=1
args.window=5
args.CNN_kernel=3
args.highway_window=3
args.alpha=True
#
# for window in [3,4,5,6,7]:
#     for highway in range(3,window+1):
for site in ['ezhou','huanggang','jingzhou','wuhan','xiaogan']:
    for model in ['ARmodel','LSTNet_alpha','LSTNet']:
        args.data='county_data/{}.txt'.format(site)
        args.model=model
        args.save='save3/{}_{}.pt'.format(site,model)
        for window in [7]:
            for highway in range(7,8):
                args.window=window
                args.highway_window=window

                args.cuda = args.gpu is not None
                if args.cuda:
                    torch.cuda.set_device(args.gpu)
                # Set the random seed manually for reproducibility.
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    if not args.cuda:
                        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
                    else:
                        torch.cuda.manual_seed(args.seed)

                Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize);
                print(Data.rse);
                model = eval(args.model).Model(args, Data);

                if args.cuda:
                    model.cuda()

                nParams = sum([p.nelement() for p in model.parameters()])
                print('* number of parameters: %d' % nParams)

                if args.L1Loss:
                    criterion = nn.L1Loss(size_average=False);
                else:
                    criterion = nn.MSELoss(size_average=False);
                evaluateL2 = nn.MSELoss(size_average=False);
                evaluateL1 = nn.L1Loss(size_average=False)
                if args.cuda:
                    criterion = criterion.cuda()
                    evaluateL1 = evaluateL1.cuda();
                    evaluateL2 = evaluateL2.cuda();


                best_val = 10000000000000000;
                optim = Optim.Optim(
                    model.parameters(), args.optim, args.lr, args.clip,
                )
                all_loss=[]
                # At any point you can hit Ctrl + C to break out of training early.
                try:
                    print('begin training');
                    for epoch in range(1, args.epochs+1):
                        epoch_start_time = time.time()
                        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                        val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size);
                        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
                        # save the model if the validation loss is the best we've seen so far.

                        if val_loss < best_val:
                            with open(args.save, 'wb') as f:
                                torch.save(model, f)
                            best_val = val_loss
                        if epoch % 5 == 0:
                            test_acc, test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size);
                        # print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
                        # all_loss.append([train_loss,val_loss,test_acc])
                    # print('当前参数',args.window,args.highway_window)
                except KeyboardInterrupt:
                    print('-' * 89)
                    print('Exiting from training early')

                # Load the best saved model.
                with open(args.save, 'rb') as f:
                    model = torch.load(f)
                test_acc, test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size);
                print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

                # print(all_loss[-1])
                # print('参数window='+str(args.window)+' ','highway='+str(args.highway_window)+'，测试集loss',all_loss[-1][-1],',训练集loss',all_loss[-1][0])
                # plt.plot(all_loss)
                # plt.show()