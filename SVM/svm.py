import os
import sys
import numpy as np
import scipy
import random

# import functions from previous assignment .py
sys.path.insert(0, os.getcwd() + "/DecisionTree") # add decision tree folder to system path
import DT_Practice as DT  

sys.path.insert(0, os.getcwd() + "/Perceptron") # add perceptron folder to system path
import Percep as per


# function to train linear SVM
def trainSVM(trainData, trainLabels, y0, yt, C, numEpochs):
    # initialize weights
    w = np.zeros(trainData.shape[1] + 1)
    t = 0
    for epoch in range(numEpochs):
        # shuffle training examples
        shuffIdx = list(range(trainData.shape[0]))
        random.shuffle(shuffIdx)
        data = trainData[shuffIdx, :]
        labels = trainLabels[shuffIdx]
        # iterate through training examples
        for exIdx in range(data.shape[0]):
            curX = data[exIdx,:]
            curX = np.append(curX, 1)
            upType = labels[exIdx] * np.dot(w, curX)
            # update weights
            if upType <= 1:
                w = (1-y0)*w + y0*C*labels[exIdx]*curX
            else:
                w = (1-y0)*w
            # update learning rate using y_t function
            t += 1
            y0 = yt(y0, t)
    # return weights
    return w


# function to do a forward pass on a trained linear SVM
def forwardLinSVM(w, data, labels):
    output = []
    for rowIdx in range(data.shape[0]): # iterate through each row in data
        curX = data[rowIdx, :]
        curX = np.append(curX, 1)
        # calculate prediction
        pred = np.dot(w, curX)
        if pred > 0: # if statement to implement sgn() function
            output.append(1)
        else:
            output.append(-1)
    # convert output array to numpy array
    output = np.asarray(output) 
    # compare output to labels to calculate accuracy
    acc = np.sum(output == labels)/labels.size

    return acc, output





        

# functions for calculating learning rate update
def y_t0(y0, t, a=100):
    y_t = y0/(1 + ((y0*t)/a))
    return y_t

def y_t1(y0, t):
    y_t = y0/(1+t)
    return y_t


# only run if script is the main script being called
if __name__ == "__main__":
    trainPath = "Data/bank-note/train.csv"
    testPath = "Data/bank-note/test.csv"
    # load in data
    trainData = DT.LoadData(trainPath)
    trainData = trainData.astype(float)
    testData = DT.LoadData(testPath)
    testData = testData.astype(float)
    # grab labels from loaded in data
    trainLabels = trainData[:, trainData.shape[1]-1]
    testLabels = testData[:, testData.shape[1]-1]
    # covert labels to 1 or -1
    trainLabels = per.convertLabel(trainLabels)
    testLabels = per.convertLabel(testLabels)
    # delete labels from data matrix
    trainData = np.delete(trainData, trainData.shape[1]-1, axis=1)
    testData = np.delete(testData, testData.shape[1]-1, axis=1)

    #### Problem 2A ####
    C = [100/873, 500/873, 700/873]
    w_ytA = []
    for k in range(len(C)):
        y0 = 0.1
        w = trainSVM(trainData, trainLabels, y0, y_t0, C[k], 100)
        # save weights for comparison later
        w_ytA.append(w)
        # calculate training and testing errors
        trainAcc, output = forwardLinSVM(w, trainData, trainLabels)
        trainError = 1 - trainAcc
        testAcc, output = forwardLinSVM(w, testData, testLabels)
        testError = 1 - testAcc
        # print errors
        print("For a C value of {} and y_t = y0/(1 + ((y0*t)/a))".format(C[k]))
        print("Training Error: {}".format(trainError))
        print("Testing Error: {}".format(testError))
        print("\n")

    #### Problem 2B ####
    w_ytB = []
    for k in range(len(C)):
        y0 = 1
        w = trainSVM(trainData, trainLabels, y0, y_t1, C[k], 100)
        # save weights for comparison later
        w_ytB.append(w)
        # calculate training and testing errors
        trainAcc, output = forwardLinSVM(w, trainData, trainLabels)
        trainError = 1 - trainAcc
        testAcc, output = forwardLinSVM(w, testData, testLabels)
        testError = 1 - testAcc
        # print errors
        print("For a C value of {} and y_t = y0/(1+t)".format(C[k]))
        print("Training Error: {}".format(trainError))
        print("Testing Error: {}".format(testError))
        print("\n") 

    #### Problem 3A ####
    