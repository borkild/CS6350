import numpy as np

# function to load in the data
def LoadData(dataPath):
    with open(dataPath,'r') as f:
        for line in f:
            terms = line.strip(' ').split(',')
