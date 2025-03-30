"""
Robert Goff
Last Updated: March 19 2025

Used to create Kdtree binning map to SMEFT data using
turtle package, and then create data sets for the ALFI
cdf NN training.
"""

import os, sys

# standard data structure packages
import numpy as np
import pandas as pd
import random as rd
# imports turtle modules for dktree binning
import turtlebinning as tt
from array import array

# pytorch interface
import torch
import torch.nn as nn

# plotting packages
import matplotlib as mp
import matplotlib.pyplot as plt

# utilites file for repository
import utils as ut

import time
# ------------------------------------------------------
def plot_distribution(y1, y2, title='plot.png', label1='SMEFT', label2='SM', filename='dist.png',
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
    plt.title(title)
    plt.xlim(xmin, xmax)
    #plt.ylim(0, 4.5)

    plt.hist(sm,
             bins=nbins,
             color='red',
             alpha=0.3,
             range=(xmin, xmax),
             density=True,
             label=label2)
    plt.legend(fontsize='small')
    plt.xlabel('Statistic value')
    plt.hist(smeft,
             bins=nbins,
             color='green',
             alpha=0.3,
             range=(xmin, xmax),
             density=True,
             label=label1)
    plt.legend(fontsize='small')

    plt.savefig(filename)
    plt.show()
# ------------------------------------------------------

def main():

    # Global varibles in main
    w = 40 #weight for compute function
    PT_Scale = 200 #Gev

    # declaring names and order of ceratin pandas dataframe keys
    features = ['pT(b1)', 'pT(l1)', 'dR(l1 l2)', 'dPhi(l1 l2)', 'dEta(l1 l2)', 'ctz', 'ctl1']
    observables = ['pT(b1)', 'pT(l1)', 'dR(l1 l2)', 'dPhi(l1 l2)', 'dEta(l1 l2)']
    theta = ['ctz', 'ctl1']
    target  = 'target'

    # update fonts
    FONTSIZE = 14
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : FONTSIZE}
    mp.rc('font', **font)

    # load in full data set and then sort by theta values
    dfFull = pd.read_csv('../data_sets/SMEFT_fulldata.csv')
    print('Data loaded: \n', dfFull[100:105])
    dfFull = dfFull.sort_values(by=theta)
    print('Data sortedby parameter space value: \n', dfFull[:4])

    # scale data to match training data
    dfFull['pT(b1)']/= PT_Scale
    dfFull['pT(l1)']/= PT_Scale

    # create new numpy array for data to be used in library
    params = dfFull[theta].to_numpy()
    params, SMEFT_idx = np.unique(params, axis=0, return_index=True)
    params = np.unique(params, axis=0)
    print('indices \n', SMEFT_idx[:5])
    print('Uniquew sorted params array: \n', params[:5])
    SMEFT_data = dfFull[features].iloc[SMEFT_idx].to_numpy()   # SMEFT data of both Observables and parameters
    D_x        = dfFull[observables].iloc[SMEFT_idx].to_numpy()# SMEFT data of only observables
    print('SMEFT data array: \n', SMEFT_data[:5])
    print('Observables only: \n', D_x[:5])
    
    # creates model object and loads the model dictionary into the object
    model = nn.Sequential(nn.Linear( 7, 20), nn.SiLU(),    # layer 0
                          nn.Linear(20, 20), nn.SiLU(),    # layer 1
                          nn.Linear(20, 20), nn.SiLU(),    # layer 2
                          nn.Linear(20,  1), nn.Tanh())    # layer 3
    modeldict = torch.load('trial_1.db')
    model.load_state_dict(modeldict)
    
    # useful info about data structures
    full_ln    = len(dfFull)
    param_ln   = len(params)
    feature_ln = len(features)

    # defining the data array to be used in turtle binning map
    data = array('d')
    data.extend(params[:,0])
    data.extend(params[:,1])

    # information about the size of the turtle binning map
    nbins   = 1000
    npoints = param_ln
    nparams = 2

    # call turtle to bin data
    ttb = tt.Turtle(data, nbins, npoints, nparams)

    N = 100
    # define some empty numpy objects to append in loop
    idx_map = np.zeros(shape=(param_ln, N), dtype=int) 
    
    # loops over parameter points and samples indices from the library
    # to then use in the statistic boosting
    for i in range(param_ln):
        
        # set param point
        x = params[i,0]
        y = params[i,1]
        point = array('d', [x, y])

        # find relavent bin and indices then sample
        j = ttb.findBin(point)
        ii = ttb.indices(j)
        jj = rd.choices(ii, k=N)

        # map random sampled indcies to the relvaent params points
        idx_map[i] = jj

        if i % 100 == 0:
            print(f'\rCurrent iteration: {i:d}', end = '')

    print()
    # make a shuffled index map for ALFI
    print('idx : \n', idx_map[:5,:5])
    idx_map_shuff = idx_map.copy()
    np.random.shuffle(idx_map_shuff)
    
    print('idx : \n', idx_map[:5,:5])
    print('idx shuff: \n', idx_map_shuff[:5,:5])
    
    # create larger array of stuff
    data = np.zeros(shape=(param_ln * N, feature_ln))
    data_shuff = np.zeros(shape=(param_ln * N, feature_ln))
    
    start = time.time()
    for i in range(param_ln):
        ii = i*N

        for j in range(N):
            data[ii+j] = np.concatenate((D_x[idx_map[i,j]], params[i]), axis=0)
            data_shuff[ii+j] = np.concatenate((D_x[idx_map_shuff[i,j]], params[i]), axis=0)
        if i % 100 == 0:
            print(f'\rCurrent interation: {i:d}', end = '')
    end = time.time()
    print()
    print('data values: \n', data[:5])
    print('time elsaped in loop: ', end-start)

    for i in range(1):
        ii = i * N
        obs_1 = D_x[idx_map[i], 1]
        obs_2 = D_x[idx_map[i], 2]
        obs_3 = D_x[idx_map[i], 3]
        print('obs_1: \n:', obs_1)
        sys.exit()
    lamb = -ut.compute(model, data, w=40)
    lamb_shuff = -ut.compute(model, data_shuff, w=40)

    lamb_n = ut.sum_stat(lamb, N=param_ln * N, M=N, ave=True)
    lamb_shuff_n = ut.sum_stat(lamb_shuff, N=param_ln * N, M=N, ave=True)

    print('N=100 Stat: \n', lamb_n[:5])
    print('N=100 Shuffled Stat: \n', lamb_shuff_n[:5])

    dfLamb = pd.DataFrame({'lamb': lamb_n, 'lamb_obs': lamb_shuff_n, 'ctz': params[:,0], 'ctl1': params[:,1]})
    dfLamb.to_csv('lamb_ALFI.csv', index=False)

main()


