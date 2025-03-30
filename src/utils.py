"""
Robert Goff, Harrison Prosper
Last updated: Dec 10, 2024

utilites script for some useful function and 
class declarations for EFT LFI
"""

import os,sys

# data manipulation
import numpy as np
import pandas as pd
import random as rd

# plotting libraries
import matplotlib.pyplot as plt
import matplotlib

# Standard pytorch libraries
import torch
import torch.nn as nn

# standard measures of model performance
from sklearn.metrics import roc_curve, auc

# imports turtle modules for dktree binning uses root
# import turtlebinning as tt
# from array import array


#--------------------------------------------------------------------
# useful functions

# compute trained statistic value by importing model
def compute(f, x, w):
    # go to evaluation mode
    f.eval()
    # 1. convert to a tensor
    # 2. compute
    # 3. reshape to a 1D tensor
    # 4. detach from computation tree and convert to a numpy array
    # 5. scale by the factor "w"
    x = torch.Tensor(x)
    y = w * f(x).view(-1,).detach().numpy()
    return y

# sum statistic without use of shower library. 
def sum_stat(stat, N=100, M=100, ave=False):
    if N < M:
        print('error: cannot have stat of more events than generated')
        return 0
    if not N % M == 0:
        print('error: stat value must be muilitple of total events')
        return 0

    # find total length of final array
    l = int(N/M)
    stat_sum = np.zeros(l)
    # checks if the final array is averaged or direct sum
    # does not effect results but must be consistant 
    if ave:
        for i in range(l):
            ii = i*M
            stat_sum[i] = stat[ii:ii+M].mean()
    else:
        for i in range(l):
            ii = i*M
            stat_sum[i] = stat[ii:ii+M].sum()

    return stat_sum

# finds target value for training cdf via NN
# must be 1d numpy arrays passed into function
def find_Z(lamb, lamb_obs):
    if len(lamb) != len(lamb_obs):
        print('arrays must be of same length')
        return 0

    target = np.zeros(len(lamb))
    for i in range(len(lamb)):
        if lamb[i] < lamb_obs[i]:
            target[i] = 1

    return target

#def shower_sim(n, m, nbins, nparams, 
#---------------------------------------------------------
# plotting functions

def plot_average_losses(losses):

    xx, yy_t, yy_v = losses

    # create an empty figure
    fig = plt.figure(figsize=(6, 5))
    fig.tight_layout()

    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)

    ax.plot(xx, yy_t, color='red',  lw=1, label='training loss')
    ax.plot(xx, yy_v, color='blue', lw=1, label='validation loss')
    ax.legend()

    ax.set_xlabel('iterations', fontsize=FONTSIZE)
    ax.set_ylabel('average loss', fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')

    plt.show()

def plot_cov(df, confLvl,
             filename = 'param_space_coverage.png',
              xmin=-5.0, xmax=5.0, xstep=0.5,
              xname='ctz', xlabel='ctz',
              ymin=-5.0, ymax=5.0, ystep=0.5,
              yname='ctl1', ylabel='ctl1',
              ftsize=18):
    region = df[df.coverage > confLvl]
    print(region)
    fig = plt.figure(figsize=(5, 5))

    # set axes' limits
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # annotate axes
    plt.xlabel(xlabel, fontsize=ftsize)
    plt.xticks(np.arange(xmin, xmax+xstep, xstep))

    plt.ylabel(ylabel, fontsize=ftsize)
    #plt.yticks(np.arange(ymin, ymax+ystep, ystep))

    plt.scatter(region[xname], region[yname], marker='o',
                s=30, c='green', alpha=0.3, label='SMEFT')

    plt.legend(loc='upper left', fontsize='small') # activate legend

    fig.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_distribution(y1, y2,
                      nbins=100,
                      xmin=-0.15,
                      xmax= 0.15,
                      ftsize=14,
                      fgsize=(6, 4)):

    # select the model output results for "SMEFT" events
    smeft = y1

    # select the model output results for "SM" events
    sm    = y2

    # set size of figure
    plt.figure(figsize=fgsize)
    plt.title("N=100 Events Sampled")
    plt.xlim(xmin, xmax)
    #plt.ylim(0, 4.5)

    plt.hist(sm,
             bins=nbins,
             color='red',
             alpha=0.3,
             range=(xmin, xmax),
             density=True,
             label='SM')
    plt.legend(fontsize='small')
    plt.xlabel('Statistic value')
    plt.hist(smeft,
             bins=nbins,
             color='green',
             alpha=0.3,
             range=(xmin, xmax),
             density=True,
             label='SMEFT')
    plt.legend(fontsize='small')

    plt.savefig("N100FullSpaceDist.png")
    plt.show()

def plot_ROC(y, p):

    bad, good, _ = roc_curve(y, p)

    roc_auc = auc(bad, good)
    plt.figure(figsize=(5, 5))
    plt.title("N=100 Events Sampled")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('fraction of SM events', fontsize=14)
    plt.ylabel('fraction of SMEFT events', fontsize=14)

    plt.plot(bad, good, color='red',
             lw=2, label='ROC curve, AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

    plt.legend(loc="lower right", fontsize=14)

    plt.savefig("N100EventsFullSpace_ROC.png")
    plt.show()
#---------------------------------------------------------------------
# training funtions

def dataframe2tensor(df, target, source, dev):
    # change from pandas dataframe to PyTorch tensors
    # and load data to device.
    x = torch.tensor(df[source].to_numpy()).float().to(dev)
    t = torch.tensor(df[target].to_numpy()).int().to(dev)
    return (x, t)

def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_batch(x, t, batch_size, n):
    i = n * batch_size # first row
    j = i + batch_size # last row + 1 # a Python oddity!
    return x[i:j], t[i:j]

def get_random_batch(x, t, batch_size, n=None):
    indices = np.random.choice(len(x), batch_size)
    return x[indices], t[indices]

def validate(model, avloss, x, t):
    # set to evaluation mode so that any training
    # specific operations are disabled.
    model.eval()

    with torch.no_grad(): # no need to compute gradients wrt. x and t
        # reshape to ensure that y and t are of the same shape!
        y = model(x).reshape(t.shape)
    return avloss(y, t)

def train_model(epoch,
                model, optimizer, averageloss, getbatch,
                train_x, train_t,
                valid_x, valid_t,
                batch_size,
                number_iterations,
                losses,
                step=10):
    import copy

    # to keep track of average losses
    xx, yy_t, yy_v = losses

    n = len(valid_x)

    if epoch < 1:
        print("%5s %10s\t%10s\t%10s" % \
              ('epoch', 'iteration', 'training', 'validation'))

    # start saving best model after the
    # following number of iterations.
    start_saving = number_iterations // 100
    best_model   = None
    min_avloss   = float('inf')
    saved = ''

    for ii in range(number_iterations):
        # set mode to training so that training specific
        # operations such as dropout are enabled.
        model.train()

        # get a batch of data
        x, t = getbatch(train_x, train_t, batch_size, ii)

        # compute the output of the model for the batch of data x
        # Note: y (the output of the model) is
        #   of shape (-1, 1), but the target tensor, t, is
        #   of shape (-1,)
        # In order for the tensor operations with y and t
        # to work correctly, it is necessary that they have the
        # same shape. We can do this with the reshape method.
        y = model(x).reshape(t.shape)

        # compute a noisy approximation of the average loss.
        # Adding a bit of noise helps the minimizer escape
        # from minima that may not optimal. It is also much
        # faster to compute the loss function and its gradient
        # with respect to its parameters using batches of
        # training data rather than the full training dataset.
        empirical_risk = averageloss(y, t)

        # use automatic differentiation to compute a
        # noisy approximation of the local gradient
        optimizer.zero_grad()       # clear previous gradients
        empirical_risk.backward()   # compute gradients

        # finally, advance one step in the direction of steepest
        # descent, using the noisy local gradient.
        optimizer.step()            # move one step

        if ii % step == 0:

            avloss_t = validate(model, averageloss,
                                train_x[:n], train_t[:n])

            avloss_v = validate(model, averageloss,
                                valid_x, valid_t)

            if len(xx) < 1:
                xx.append(0)
                print("%5d %10d\t%10.6f\t%10.6f %s" % \
                      (epoch, xx[-1], avloss_t, avloss_v, saved))
            else:
                xx.append(xx[-1] + step)
                print("\r%5d %10d\t%10.6f\t%10.6f %s" % \
                      (epoch, xx[-1], avloss_t, avloss_v, saved),
                      end='')
            #moves data back to cpu after training    
            yy_t.append(float(avloss_t))
            yy_v.append(float(avloss_v))

            if avloss_v < min_avloss:
                min_avloss = avloss_v

                if ii > start_saving:
                    saved = str(ii)
                    best_model = copy.deepcopy(model)

    print()
    return (xx, yy_t, yy_v), best_model
#---------------------------------------------------------------------
# loss functions

def average_exponential_loss(f, t, w=1):
    # f and t must be of the same shape
    losses = torch.exp(-w*t*f/2)
    return torch.mean(losses)

def average_quadratic_loss(f, t):
    # f and t must be of the same shape
    losses = (f - t)**2
    return torch.mean(losses)

def average_cross_entropy_loss(f, t):
    # f and t must be of the same shape
    # Note: because of our use of the "where" function, the
    # precise values of the targets doesn't matter so long as for
    # one class t < 0.5 and the other t > 0.5
    losses = -torch.where(t > 0.5, torch.log(f), torch.log(1 - f))
    return torch.mean(losses)
