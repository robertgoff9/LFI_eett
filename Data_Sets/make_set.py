#! /bin/env/python3
"""
Robert Goff  Last updated: Feb 20 2024

python3 MakeSet.py [SMEFT data directory] [SM data file] [identifier for new data set]

This program is  made to read in csv files and create a data set of SMEFT events.
It takes as arguments the location of the SMEFT data, the SM data file, and the identifier
for the final set of data to be created. It will then pick a specifed number of lines
from each SMEFT data file and use them to create a new data set sampled over the paremeter
space of wilson coefficents. It then combines the SMEFT data with SM data and prepares them
for use in training of a neural network.



"""
#--------------------------------------------------------------
import sys, os

import numpy as np

from glob import glob

import pandas as pd

from time import time, sleep

#--------------------------------------------------------------
def main():

    	# Name of final data set. DO NOT include file extension in this string
	SMEFT_dir = sys.argv[1]
	SM_file   = sys.argv[2]
	data_name = sys.argv[3]

	SMEFT_location = SMEFT_dir + '/*.csv'
	print(SMEFT_location)
	filenames = glob(SMEFT_location)

	# n is number of files m is the lines per file to pull
	n      = 100000
	m      = 1
	offset = 10
	print('Number of parameter points:', n)
	N = n * m
	# sets the name for the SMEFT data file to be created and opens the file for writing
	SMEFT_name    = data_name + '_SMEFT.csv'
	data_set_file = open(SMEFT_name, 'w')
	data_set_file.write('ctz,ctl1,pT(b1),pT(l1),dR(l1 l2),dPhi(l1 l2),dEta(l1 l2),WT\n')

	start = time()

	# Loops over 'n' csv files in the specified directory. Takes 'm' lines
	# from each and uses them to create a data set for training
	for i in range(n):
		recs = open(filenames[i]).readlines()

		if len(recs) == 101:
			for j in range(1, m+1):
				data_set_file.write(recs[j+offset])
		else:
			print('Check file:', filenames[i])
		if i % 100 == 0:
			print(f'\r{i:d}', end = '')

	print('\nTime elapsed to make SMEFT data:', time()-start)
	data_set_file.close()

	# Reads in SMEFT data created in the loop above. Adds target values to the data
	# and then prints the data for inspection
	SMEFT = pd.read_csv(SMEFT_name)[:N]
	SMEFT['target'] = np.ones(len(SMEFT))
	print(len(SMEFT))
	print('SMEFT dataframe:')
	print(SMEFT[:5])

	# Takes the parameter space values ( in this case a set of wilson coefficents)
	# and shuffles them to append onto the SM data
	params = SMEFT[['ctz', 'ctl1']].to_numpy()
	np.random.shuffle(params)
	params = params.T # transpose array
	params.shape
	print(len(params[0]))

	# Loads SM data to be used. Saves SM data used for final data file
	SM = pd.read_csv(SM_file)[:N]
	SM_name = data_name + '_SM.csv'
	SM.to_csv(SM_name, index=False)

	# Appends SM data with target values and parameter values (Wilson coeffients)
	SM['ctz']    = params[0]
	SM['ctl1']   = params[1]
	SM['target'] = -np.ones(len(SM))
	print('SM dataframe:')
	print(SM[:10])

	# Saves data to final csv file to be fed into training program.
	final_name = data_name + '.csv'
	df = pd.concat([SM, SMEFT]).sample(frac=1)
	print('Final dataframe:')
	print(df[:10])
	df.to_csv(final_name, index=False)

main()
