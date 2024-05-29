"""
Harrison Prosper, Robert Goff
Last Updated: May 22, 2024

python3 train_stat.py [Data file] [Identifier for resulting stat]

This program is designed to take in data generated in Sherpa or other particle
simulation software based on Dim 6
SMEFT and train a deep learning neural network to differentiate between the 
SMEFT events and the SM events generated. For the puropese of this program 
we limit the number of Wilson coeffeicents to 2 with the ability to increace 
this amount at a later date. Ensure that the SMEFT data set is seperate from
the SM data set and that both are of the format .csv
"""
# standard system modules
import os, sys

# standard module for tabular data
import pandas as pd

# standard module for array manipulation
import numpy as np

# standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt


# a truly outstanding symbolic algebra module
import sympy as sm

# widely used machine learning toolkit developed by FaceBook
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split

# standard measures of model performance
from sklearn.metrics import roc_curve, auc

# to reload modules
import importlib
#-------------------------------------------------------------
# update fonts in plots
FONTSIZE = 14
font = {'family' : 'serif',
         'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set usetex = False if LaTex is not 
# available on your system or if the 
# rendering is too slow
mp.rc('text', usetex=True)
    
# check for avalible cpu/gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Available device: {str(device):4s}')

# set parameters to be used in training 
W        = 40       # scale factor in loss function
PT_SCALE = 200      # (GeV) scales the PT values so that they are normalized
Wilson_Scale = 1    #
Ang_Scale = 1
N        = 1000000

# sets the file identifeier from the command line arguments 
ID = sys.argv[2]
print(torch.__version__)

#-----------------------------------------------------------
# Plotting functions

# Plot the parameter space of the data set. In this case should be 
# uniform across (ctz, ctl1)
XMIN = -5.0
XMAX =  5.0
XSTEP=  0.5
XNAME= 'ctz'
XLABEL = r'$ct_z$'

YMIN = -5.0
YMAX =  5.0
YSTEP=  0.5
YNAME= 'ctl1'
YLABEL = r'$ct_{l_1}$'

# plots the distribution of wilson coeffectients in param space
def plot_data(df, 
              filename = ID + '_param_space.png',
              xmin=XMIN, xmax=XMAX, xstep=XSTEP, 
              xname=XNAME, xlabel=XLABEL,
              ymin=YMIN, ymax=YMAX, ystep=YSTEP,
              yname=YNAME, ylabel=YLABEL,
              ftsize=18):
    
    smeft = df[df.target > 0.5]
    
    sm    = df[df.target < 0.5]
    
    # set size of figure
    fig = plt.figure(figsize=(5, 5))

    # set axes' limits
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    # annotate axes
    plt.xlabel(xlabel, fontsize=ftsize)
    plt.xticks(np.arange(xmin, xmax+xstep, xstep))
    
    plt.ylabel(ylabel, fontsize=ftsize)
    plt.yticks(np.arange(ymin, ymax+ystep, ystep))

    plt.scatter(smeft[xname], smeft[yname], marker='o',
                s=30, c='green', alpha=0.3, label='SMEFT')
    
    plt.scatter(sm[xname], sm[yname], marker='*',
                s=80, c='red',  alpha=0.4, label='SM')
    
    plt.legend(loc='upper left', fontsize='small') # activate legend
    
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()

# plots the losses of the model training 
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

# plots a histogram of SMEFT and SM values for the statistic
def plot_distribution(t, y,
                      filename = ID + '_stat_distribution.png',
                      nbins=100, 
                      xmin=-1, 
                      xmax= 1 , 
                      ftsize=14, 
                      fgsize=(6, 4)):

    # select the model output results for "SMEFT" events
    smeft = y[t > 0.5]
    
    # select the model output results for "SM" events
    sm    = y[t < 0.5]
    
    # set size of figure
    plt.figure(figsize=fgsize)
    
    #plt.xlim(xmin, xmax)
    #plt.ylim(0, 1.5)
    
    plt.hist(sm, 
             bins=nbins, 
             color='red',
             alpha=0.3,
             range=(xmin, xmax), 
             density=True, 
             label='SM') 
    plt.legend(fontsize='small')
    
    plt.hist(smeft, 
             bins=nbins, 
             color='green',
             alpha=0.3,
             range=(xmin, xmax), 
             density=True, 
             label='SMEFT')
    plt.legend(fontsize='small')

    plt.savefig(filename)
    plt.show()

def plot_ROC(y, p, filename = ID + '_ROC.png'):
    
    bad, good, _ = roc_curve(y, p)
    
    roc_auc = auc(bad, good)
    plt.figure(figsize=(5, 5))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('fraction of SM events', fontsize=14)
    plt.ylabel('fraction of SMEFT events', fontsize=14)
    
    plt.plot(bad, good, color='red',
             lw=2, label='ROC curve, AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

    plt.legend(loc="lower right", fontsize=14)
    
    plt.savefig(filename)
    plt.show()

#---------------------------------------------------------------------
# Functions for the neural network
# Note: there are several average loss functions available 
# in PyTorch, such as nn.CrossEntropyLoss(), but it's useful 
# to know how to create your own.

def average_exponential_loss(f, t, w=W):
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
                
            yy_t.append(avloss_t)
            yy_v.append(avloss_v)
            
            if avloss_v < min_avloss:
                min_avloss = avloss_v
                
                if ii > start_saving:
                    saved = str(ii)
                    best_model = copy.deepcopy(model)

    print()      
    return (xx, yy_t, yy_v), best_model


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

#---------------------------------------------------------------------------

def dataframe2tensor(df, target, source):
    # change from pandas dataframe to PyTorch tensors
    # and load data to device.
    x = torch.tensor(df[source].to_numpy()).float().to(device)
    t = torch.tensor(df[target].to_numpy()).int().to(device)
    return (x, t)


class Model(nn.Module):
    
    def __init__(self, n_inputs=7, n_nodes=20, n_layers=2):

        # call constructor of base (or super, or parent) class
        super(Model, self).__init__()

        self.n_inputs = n_inputs
        self.n_nodes  = n_nodes
        self.n_layers = n_layers
        
        # create input layer
        self.layer0 = nn.Linear(n_inputs, n_nodes)
        
        # cache layers in a list for later use
        self.layers = []
        self.layers.append(self.layer0)

        # create "hidden" layers
        for l in range(1, n_layers):
            cmd = 'self.layer%d = nn.Linear(%d, %d)' % \
            (l, n_nodes, n_nodes)
            exec(cmd)
            cmd = 'self.layers.append(self.layer%d)' % l
            exec(cmd)
          
        # create output layer
        cmd = 'self.layer%d = nn.Linear(%d, 1)' % (n_layers, n_nodes)
        exec(cmd)
        cmd = 'self.layers.append(self.layer%d)' % n_layers
        exec(cmd)

    # define (required) method to compute output of network
    def forward(self, x):
        y = x
        for layer in self.layers[:-1]:
            y = layer(y)
            y = torch.layer_norm(y, [self.n_nodes])
            y = torch.sin(y)
        y = self.layers[-1](y)
        return y

def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#---------------------------------------------------------------------------

def main():
    
    FONTSIZE = 14
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : FONTSIZE}
    mp.rc('font', **font)

    # set usetex = False if LaTex is not 
    # available on your system or if the 
    # rendering is too slow
    mp.rc('text', usetex=True)
    
    # check for avalible cpu/gpu
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Available device: {str(device):4s}')

    # load in data to dataframe using command line argument
    df = pd.read_csv(sys.argv[1])
    df['pT(b1)'] /= PT_SCALE
    df['pT(l1)'] /= PT_SCALE
    #df['ctz']    /= Wilson_Scale
    #df['ctl1']   /= Wilson_Scale
    #df['dEta(l1 l2)'] /= Ang_Scale
    #df['dPhi(l1 l2)'] /= Ang_Scale
    #df['dR(l1 l2)'] /= Ang_Scale
    print(len(df))
    print(df[:10])

    # defines different columns in data frame for easy use
    features= ['pT(b1)', 'pT(l1)', 'dR(l1 l2)', 'dPhi(l1 l2)', 'dEta(l1 l2)', 'ctz', 'ctl1']
    target  = 'target'

    # here we split the data into training, testing, and validation sets
    fraction= 1/20
    M       = int(len(df) * fraction)
    test    = df[:M]
    train   = df[M:]

    print(len(train), len(test))

    print(train[:10])

    # Split the training data into a part for fitting and
    # a part for validation during training.
    fraction = 1/19
    train_data, valid_data = train_test_split(train, test_size=fraction)

    # reset the indices in the dataframes (and drop the old ones)
    # so that the indices start from zero and increment by one
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    test_data  = test

    # save data sets for future reference 
    train_data.to_csv(ID + '_train.csv', index=False)
    valid_data.to_csv(ID + '_valid.csv', index=False)
    test_data.to_csv(ID + '_test.csv',   index=False)

    print('train set size:        %6d' % train_data.shape[0])
    print('validation set size:   %6d' % valid_data.shape[0])
    print('test set size:         %6d' % test_data.shape[0])

    # transforms the objects to pytorch tensors for training
    train_x, train_t = dataframe2tensor(train_data, target, features)
    valid_x, valid_t = dataframe2tensor(valid_data, target, features)
    test_x,  test_t  = dataframe2tensor(test_data,  target, features)

    #print(train_x[:5], train_t[:5])

    k = 2000
    plot_data(train[:k])
    #print(len(features))

    # defines the model mannually, alternitively you could use the model class 
    # defined above. If you use the mannual method you must make sure this is 
    # defined the same in subseqent programs
    model = nn.Sequential(nn.Linear( 7, 20), nn.SiLU(),    # layer 0
                          nn.Linear(20, 20), nn.SiLU(),    # layer 1
                          nn.Linear(20, 20), nn.SiLU(),    # layer 2
                          nn.Linear(20,  1), nn.Tanh())    # layer 3
    
    #writefile teststatistic.py


    # load model
    import teststatistic as ts
    importlib.reload(ts)
    #model = ts.Model()

    print(model)
    print('number of parameters: %d' % number_of_parameters(model))

    # defines aspects of the model to be trained
    average_loss  = average_exponential_loss
    step          = 1
    losses        = ([], [], [])

    niterations   = 20000
    batch_size    = 100   # sample over which to compute average loss
    learning_rate = 1.e-3
    weight_decay  = 0.0
    optimizer     = torch.optim.Adam(model.parameters(), 
                                     lr=learning_rate, 
                                     weight_decay=weight_decay) 

    losses, best_model = train_model(1, 
                                     model, optimizer, average_loss, 
                                     get_random_batch,
                                     train_x, train_t, 
                                     valid_x, valid_t,
                                     batch_size, niterations,
                                     losses, step)
    
    # save best model parameters
    torch.save(best_model.state_dict(), ID + '.db')
    plot_average_losses(losses)

    test_y = compute(best_model, test_x)
    plot_distribution(test_t, test_y)
    plot_ROC(test_t, test_y)

main()
