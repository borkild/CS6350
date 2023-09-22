# this script is to create a dummy submission for the class kaggle competition
import numpy as np
import os
import sys
import csv

# function to load in the data
def LoadData(dataPath):
    with open(dataPath,'r') as f:
        dataArray = f.readlines() # read lines from file into list
        numLine = len(dataArray) # get number of lines in csv file
        numAtt = len(dataArray[0].strip(' ').split(',')) # get number of attributes
        data = np.empty((numLine,numAtt),dtype=np.dtype('U100')) # create numpy array to write data to
        for lineIdx in range(len(dataArray)): # iterate through list and break data up
            terms = dataArray[lineIdx].strip(' ').split(',')
            term_count = 0
            for val in terms:
                data[lineIdx,term_count] = val.strip('\n') # get rid of new line '\n'
                term_count += 1
    return data

tstData = LoadData('FinalProject/data/test_final.csv') # load in test data

datShape = np.shape(tstData) # find shape of data

subData = np.zeros((datShape[0]-1,2),dtype = int)

subData[:,0] = tstData[1:datShape[0],0].astype(int) # get IDs

subData[:,1] = 0 # make all predictions 0

dummyFile = "FinalProject/Submissions/Orkild_DummySubmission.csv"

fields = ['ID', 'Prediction']

# write data to csv file
with open(dummyFile, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(subData)






