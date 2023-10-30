import os
import sys
import numpy as np

# now import functions from custom decision tree implementation
# we assume our function is called from the CS6350 folder, which should occur if we run from the run script
sys.path.insert(0, os.getcwd() + "/DecisionTree") # add decision tree folder to system path
import DT_Practice as DT

# function to convert labels from 0 and 1 to -1 and 1
def convertLabel(labels):
    zeroIdx = np.argwhere(labels == 0)
    labels[zeroIdx] = -1
    return labels


# function to generate standard perceptron weights
def trainPerceptron(trainData, numEpochs, r):
    pass




# function to do forward pass through perceptron with weights
def percepForward():
    pass


if __name__ == "__main__":
    # pathes to data
    trainDataPath = "Data/bank-note/train.csv"
    testDataPath = "Data/bank-note/test.csv"
    # load in data and separate labels
    trainData = DT.LoadData(trainDataPath)
    trainData = trainData.astype(float)
    trainLabels = trainData[:,trainData.shape[1]-1] # grab training labels
    trainLabels = convertLabel(trainLabels) # convert labels from 1 or 0 to 1 or -1
    trainData = np.delete(trainData, trainData.shape[1]-1, axis=1) # delete labels from data array


    testData = DT.LoadData(testDataPath)
    testData = testData.astype(float)
    testLabels = testData[:, testData.shape[1]-1] # grab testing labels
    testLabels = convertLabel(testLabels) # convert labels from 1 or 0 to 1 or -1
    testData = np.delete(testData, testData.shape[1]-1, axis=1) # delete labels from training array

    # now we apply the perceptron algorithm to our data


    ben = 1
    
    






