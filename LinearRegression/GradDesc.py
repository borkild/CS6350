import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import random as rand

# now import functions from custom decision tree implementation
# we assume our function is called from the CS6350 folder, which should occur if we run from the run script
sys.path.insert(0, os.getcwd() + "/DecisionTree") # add decision tree folder to system path
import DT_Practice as DT 




if __name__ == "__main__":
    trainData = "Data/concrete/train.csv"
    data = DT.LoadData(trainData)


    ben = 1
