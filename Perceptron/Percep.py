import os
import sys
import random
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
def trainPerceptron(trainData, trainLabels, numEpochs = 100, learnRate = 1.0):
    #initialize weights, we include the bias in our weight vector, so we add 1 additional parameter to it
    w = np.zeros(trainData.shape[1] + 1)
    # iterate through each epoch
    for epoch in range(numEpochs):
        # shuffle data and labels
        shuffIdx = list(range(trainData.shape[0]))
        random.shuffle(shuffIdx)
        data = trainData[shuffIdx,:]
        labels = trainLabels[shuffIdx]
        # iterate through training examples
        for exIdx in range(data.shape[0]):
            x_i = np.append(data[exIdx, :], 1) # current row of data
            y_i = labels[exIdx] # current label
            updateW = np.dot(w*y_i, x_i)
            if updateW <= 0: # update weights 
                w = w + learnRate*y_i*x_i
    
    # return weights and accuracies
    return w

# function to train voted perceptron
def trainVotedPercep(trainData, trainLabels, numEpochs = 100, learnRate = 1.0):
    #initialize weights, we include the bias in our weight vector, so we add 1 additional parameter to it
    w = np.zeros(trainData.shape[1] + 1)
    # initialize list to track w and c
    wList = []
    cList = []
    # iterate through each epoch
    for epoch in range(numEpochs):
        # shuffle data and labels
        shuffIdx = list(range(trainData.shape[0]))
        random.shuffle(shuffIdx)
        data = trainData[shuffIdx,:]
        labels = trainLabels[shuffIdx]
        c = 1
        # iterate through training examples
        for exIdx in range(data.shape[0]):
            x_i = np.append(data[exIdx, :], 1) # current row of data
            y_i = labels[exIdx] # current label
            updateW = np.dot(w*y_i, x_i)
            if updateW <= 0: # update weights 
                w = w + learnRate*y_i*x_i
                wList.append(w)
                cList.append(c)
                c = 1
            else:
                c = c + 1
    
    # return weights and accuracies
    return wList, cList


# function to train an average perceptron
def trainAveragePercep(trainData, trainLabels, numEpochs = 100, learnRate = 1.0):
    #initialize weights and average, we include the bias in our weight vector, so we add 1 additional parameter to it
    w = np.zeros(trainData.shape[1] + 1)
    a = np.zeros(trainData.shape[1] + 1)
    # iterate through each epoch
    for epoch in range(numEpochs):
        # shuffle data and labels
        shuffIdx = list(range(trainData.shape[0]))
        random.shuffle(shuffIdx)
        data = trainData[shuffIdx,:]
        labels = trainLabels[shuffIdx]
        # iterate through training examples
        for exIdx in range(data.shape[0]):
            x_i = np.append(data[exIdx, :], 1) # current row of data
            y_i = labels[exIdx] # current label
            updateW = np.dot(w*y_i, x_i)
            if updateW <= 0: # update weights and average
                w = w + learnRate*y_i*x_i
                a = a + w
    
    # return average, a
    return a



# function to do forward pass through perceptron with weights
def percepStandForward(w, data, labels):
    output = np.zeros(labels.shape)
    for exIdx in range(data.shape[0]): # iterate through data rows
        x_i = np.append(data[exIdx,:], 1) # grab current example
        out = np.dot(w, x_i) # calculate output from model
        if out < 0: # update output array based on sign of out
            output[exIdx] = -1
        else:
            output[exIdx] = 1

    acc = np.sum(labels == output)/labels.size

    return output, acc
        
# function to do forward pass on voted perceptron
def percepVotedForward(w, c, data, labels):
    output = np.zeros(labels.shape)
    for exIdx in range(data.shape[0]): # iterate through data rows
        out = 0
        for wIdx in range(len(w)): # iterate through weight vectors
            x_i = np.append(data[exIdx,:], 1) # grab current example
            w_i = w[wIdx]
            out = np.dot(w_i, x_i)*c[wIdx] + out # calculate output from model
        if out < 0: # update output array based on sign of out
            output[exIdx] = -1
        else:
            output[exIdx] = 1

    acc = np.sum(labels == output)/labels.size

    return output, acc


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

    # now we apply the standard perceptron algorithm to our data
    trainErrorST = []
    testErrorST = []
    for numEpoch in range(1,11):
        weights = trainPerceptron(trainData, trainLabels, numEpoch, learnRate=0.1)
        out, trainAcc = percepStandForward(weights, trainData, trainLabels)
        trainErrorST.append(1 - trainAcc)
        out, testAcc = percepStandForward(weights, testData, testLabels)
        testErrorST.append(1 - testAcc)



    # apply voted perceptron
    trainErrorV = []
    testErrorV = []
    for numEpoch in range(1,11):
        weights, c = trainVotedPercep(trainData, trainLabels, numEpoch, learnRate=0.1)
        out, trainAcc = percepVotedForward(weights, c, trainData, trainLabels)
        trainErrorV.append(1 - trainAcc)
        out, testAcc = percepVotedForward(weights, c, testData, testLabels)
        testErrorV.append(1 - testAcc)
    
    # apply average perceptron


ben = 1



