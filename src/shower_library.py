"""
Robert Goff, Harrison Prosper
"""


# standard system modules
import os, sys

# standard module for tabular data
import pandas as pd

# standard module for array manipulation
import numpy as np
import random as rd

#imports turtle modules for dktree binning
import turtlebinning as tt
from array import array

# standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt

# widely used machine learning toolkit developed by FaceBook
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split

# standard measures of model performance
from sklearn.metrics import roc_curve, auc

# to reload modules
import importlib

#loads in trained model
import teststatistic as ts

#-------------------------------------------------------------------------
# update fonts
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set usetex = False if LaTex is not 
# available on your system or if the 
# rendering is too slow
mp.rc('text', usetex=True)

# set a seed to ensure reproducibility
seed = 128
rnd  = np.random.RandomState(seed)
W    = 40
PT_Scale = 200 # GeV


# loads in test data and defines column values to be used
test_data = pd.read_csv('../Data_Sets/test3.csv')
print(len(test_data))

features= ['pT(b1)', 'pT(l1)', 'dR(l1 l2)', 'dPhi(l1 l2)', 'dEta(l1 l2)', 'ctz', 'ctl1']
target  = 'target'

# seperates the test data into one data frame for the SMEFT points and one for SM
# of equal length 'M' then converts to numpy
#----------------------------------------------------------------------------------

def plot_distribution(y1, y2, 
                      nbins=100, 
                      xmin=-0.5, 
                      xmax= 0.5, 
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

def compute(f, x, w=W):
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

#----------------------------------------------------------------------------------
def main():
    M = 100000
    dfSMEFT = test_data[test_data[target]>0][:M]
    dfSM    = test_data[test_data[target]<0][:M]

    dfSMEFT['pT(b1)']/= PT_Scale
    dfSMEFT['pT(l1)']/= PT_Scale
    dfSM['pT(b1)']/= PT_Scale
    dfSM['pT(l1)']/= PT_Scale
    print(len(dfSMEFT))
    print(dfSMEFT[:10])

    # converts the test data to numpy array to be used later
    test_x = test_data[features].to_numpy()
    print(test_x[:10])

    test_t = test_data[target].to_numpy()
    print(test_t[:10])

    # creates model object and loads the model dictionary into the object
    model = nn.Sequential(nn.Linear( 7, 20), nn.SiLU(),    # layer 0
                          nn.Linear(20, 20), nn.SiLU(),    # layer 1
                          nn.Linear(20, 20), nn.SiLU(),    # layer 2
                          nn.Linear(20,  1), nn.Tanh())    # layer 3
    modeldict = torch.load('../Model_1/teststatistic.db')
    model.load_state_dict(modeldict)

    print(test_x.shape)




    test_y = compute(model, test_x)

    print(test_y[:10])


    #plot_distribution(test_y, test_t)

    # Creates and plots ROC and AOC based on data and target values
    #plot_ROC(test_t, test_y)


    #SMEFT = pd.read_csv('SMEFT1.csv')


    # defining the data array to be used in turtle
    data1 = array('d')
    data1.extend(dfSMEFT['ctz'])
    data1.extend(dfSMEFT['ctl1'])

    data2 = array('d')
    data2.extend(dfSM['ctz'])
    data2.extend(dfSM['ctl1'])


    nbins   = 1000
    npoints = len(dfSMEFT)
    nparams = 2

    print(npoints)

    #print(data[:10])

    ttb1 = tt.Turtle(data1, nbins, npoints, nparams)
    ttb2 = tt.Turtle(data2, nbins, npoints, nparams)

    # N is the 'number of events' to be modeled
    # n is number of entries in the final array
    N = 1
    n = len(dfSMEFT)
    test_y1 = np.zeros(shape=(n,3))
    test_y2 = np.zeros(shape=(n,3))
    test_y3 = np.zeros(shape=(n,3))
    test_t1 = np.zeros(n)
    test_t2 = np.zeros(n)
    test_t3 = np.zeros(n)

    params = dfSMEFT[['ctz', 'ctl1']].to_numpy()
    SM = dfSM[features].to_numpy()
    #SM_obs = dfSM[['pT(b1)', 'pT(l1)', 'dR(l1 l2)', 'dPhi(l1 l2)', 'dEta(l1 l2)']].to_numpy()
    SMEFT = dfSMEFT[features].to_numpy()
    SMEFT_obs = dfSMEFT[['pT(b1)', 'pT(l1)', 'dR(l1 l2)', 'dPhi(l1 l2)', 'dEta(l1 l2)']].to_numpy()
    np.random.shuffle(SMEFT_obs)
    #print(SMEFT_obs.shape())
    SMEFT_obs = np.concatenate((SMEFT_obs, params), axis =1)
    print(SMEFT[:5])
    print(SMEFT_obs[:5])

    # loops over entries in the dataframe to get parameter values. Then finds the bin 
    # these parameters resdie in
    for i in range(n):
        x = dfSMEFT['ctz'].iloc[i]
        y = dfSMEFT['ctl1'].iloc[i]
        point = array('d', [x, y])
    
        j1 = ttb1.findBin(point)
        ii1 = ttb1.indices(j1)
        jj1 = rd.choices(ii1, k=N)

        j2 = ttb2.findBin(point)
        ii2 = ttb2.indices(j2)
        jj2 = rd.choices(ii2, k=N)
    
        # creates numpy arrays for the data to feed into compute
        #print(dfSM[features].iloc[jj2])
        #print(dfSMEFT[features].iloc[jj1])
        #sys.exit()
        #SM = dfSM[features].iloc[jj2].to_numpy()
        #SMEFT = dfSMEFT[features].iloc[jj1].to_numpy()
    
        y1 = compute(model, SMEFT[jj1])
        test_y1[i,0] = y1.mean()
        test_y1[i,1] = x
        test_y1[i,2] = y
        test_t1[i] = 1
    
        y2 = compute(model, SM[jj2])
        test_y2[i,0] = y2.mean()
        test_y2[i,1] = x
        test_y2[i,2] = y
        test_t2[i] = -1
    
    
        y3 = compute(model, SMEFT_obs[jj1])
        test_y3[i,0] = y3.mean()
        test_y3[i,1] = x
        test_y3[i,2] = y
        test_t3[i] = 1
    
        if i % 10 == 0:
            print(f'\r{i:d}', end = '')
    print("\nout of loop")
    print(test_y1[:10])
    print(test_y3[:10])
    # combined results for SM and SMEFT model computations
    test_ytot = np.append(test_y1, test_y2, axis=0)
    test_ttot = np.append(test_t1, test_t2)

    plot_distribution(test_y1[:,0], test_y2[:,0])

    plot_ROC(test_ttot, test_ytot[:,0])

    lamb = pd.DataFrame({'lamb': test_y1[:,0], 'ctz': test_y1[:,1], 'ctl1': test_y1[:,2], 'target': test_t1[:]})
    lamb_obs = pd.DataFrame({'lamb': test_y3[:,0], 'ctz': test_y3[:,1], 'ctl1': test_y3[:,2], 'target': np.zeros(n)})
    lamb.to_csv('lamb.csv', index=False)
    lamb_obs.to_csv('lamb_obs.csv', index=False)


    print(lamb[:10])
    print(lamb_obs[:10])

main()
