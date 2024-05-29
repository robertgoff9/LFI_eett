import os, sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def main():
	dflamb = pd.read_csv('lamb.csv')
	dflamb_obs = pd.read_csv('lamb_obs.csv')
	
	lamb = dflamb.to_numpy()
	lamb_obs = dflamb_obs.to_numpy()
	
	target = np.zeros(len(lamb))
	
	for i in range(len(lamb)):
		if lamb[i,0] < lamb_obs[i,0]:
			target[i] = 1
	
	print(target[:100])

main()
