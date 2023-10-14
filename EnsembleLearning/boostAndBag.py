import numpy as np
import os
import sys

# now import functions from custom decision tree implementation
# we assume our function is called from the CS6350 folder, which should occur if we run from the run script
sys.path.insert(0, os.getcwd() + "/DecisionTree") # add decision tree folder to system path
import DT_Practice as DT

# function to generate nTrees with depth of 2 based on info gain of training data
def generateTreeStumps(trainData, nTrees): 
    pass



# import bank data
