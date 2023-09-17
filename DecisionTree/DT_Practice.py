import numpy as np
import os


# function to load in the data
def LoadData(dataPath):
    with open(dataPath,'r') as f:
        dataArray = f.readlines() # read lines from file into list
        numLine = len(dataArray) # get number of lines in csv file
        numAtt = len(dataArray[0].strip(' ').split(',')) # get number of attributes
        data = np.empty((numLine,numAtt),dtype=str) # create numpy array to write data to
        for lineIdx in range(len(dataArray)): # iterate through list and break data up
            terms = dataArray[lineIdx].strip(' ').split(',')
            data[lineIdx,:] = terms
    return data

# function to execute ID3 algorithm
def ID3(data,max_depth): 
    #Note: We assume the data is a numpy array of strings, with the corresponding label in the final column
    pass 



# load in training data
CarTrainPath = 'data/car/train.csv' # assuming car data is in the same folder as script
CarTrainData = LoadData(CarTrainPath)

data = np.copy(CarTrainData)


