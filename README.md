This repository is meant to demonstrate the framework for using LFI 
methods in the case of a simulated ee -> tt process in Dim 6 SMEFT
using two wilson coeffcients. No Data sets have been included but an
example trainded statistic has been created. For more information about 
LFI methods and simulation inference in particle physics please see the 
pdf. 


---------------------------------------------------------------------
Three Main Directories:

1. Model_1: For training of the initial statistic defined as the likelihood 
   ratio. Also contains sample ROC and Distributions 

2. Model_2: For training of the second nerual network for creating the 
   the coverage set

3. Shower_library: For the use of the shower library methods of increasing
   the ability of the statistic discrimination

4. Data_Sets: Storage of data sets and scripts to process data into standard
   format.

----------------------------------------------------------------------
Required outside materials:

1. Sherpa2.2 particle generation or equivalent. 
   Whatever particle generator you use make sure that it can loop through 
   different wilson coeff properly. In the eaxmple provided 

2. Rivet HEP analysis tools. The specific scripts and analysies used are 
   provided


