import os
import sys
import numpy as np
import scipy.optimize as opt
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


# function to train SVM with kernel K
def trainKernSVM(traindata, trainLabels, K, C):
    y = np.expand_dims(trainLabels, axis=1)
    # form optimization problem and use scipy's minimization function to solve
    # define constraint
    cons = ({'type': 'eq', 'fun': constraintFunc, 'args': (y, K)})
    # define initial guess -- always start in center of bounds
    alpha_i = np.ones(trainLabels.size)*C*0.5
    # define bounds for alpha -- should be between 0 and C
    bd = np.ones((alpha_i.size, 2)) # need to define bounds for each entry in alpha vector
    bd[:,0] = 0
    bd[:,1] = C
    bd = tuple(map(tuple, bd))
    # use scipy.optimize.minimize to solve the minimization problem with the bounds, constraints, and initial guess
    # args is used to pass the value of y and k to the dualForm function
    result = opt.minimize(dualForm, alpha_i, args=(y, K), method='SLSQP', bounds=bd, constraints=cons)
    print(result.message)
    # calculate b
    alpha = result.x
    alphaStarIdx = np.argwhere((alpha > 0) & (alpha < C))
    b = 0
    for bIdx in alphaStarIdx:
        b = trainLabels[bIdx] - np.sum(alpha*trainLabels*K[:, bIdx]) + b

    b = b/alphaStarIdx.size
    # return weight vector (alpha values that minimize our function)
    return alpha, b
    


def constraintFunc(alpha, y, k):
    alpha = np.expand_dims(alpha, axis=1) # need to expand alpha's dimensions to make the matrix multiplication work right
    return np.sum(alpha*y)


# function for dual form of SVM
def dualForm(alpha, y, k):
    alpha = np.expand_dims(alpha, axis=1) # need to expand alpha's dimensions to make the matrix multiplication work right
    return 0.5*np.sum(np.matmul(y*alpha, np.transpose(y*alpha))*k) - np.sum(alpha)

# linear kernel -- form matrix of kernel for faster optimization
def linKern(data, opts = 'train', testData=None):
    if opts == 'train':
        k = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]): # there's a better way to do this, just use this for now
            for j in range(data.shape[0]):
                k[i,j] = np.dot(data[i,:],data[j,:])
        return k 
    elif opts == 'test':
        k = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            k[i] = np.dot(data[i,:], testData)
        return k
    else:
        print("opts not recognized, choose train or test")
        k = []
        return k


# gaussian kernel
def gaussKern(data, c = 1, opts = 'train', testData=None):
    if opts == 'train':
        k = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]): # there's a better way to do this, just use this for now
            for j in range(data.shape[0]):
                k[i,j] = np.exp(np.sum(np.power(data[i,:] - data[j,:], 2))/(-1*c))
        return k 
    elif opts == 'test':
        k = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            k[i] = np.exp(np.sum(np.power(data[i,:] - testData, 2))/(-1*c))
        return k
    else:
        print("opts not recognized, choose train or test")
        k = []
        return k


# function to run forward pass through trained SVM with kernel, K
def forwardKernSVM(alpha, b, K, gamma, testData, testLabels, trainData, trainLabels):
    output = []
    for testIdx in range(testData.shape[0]):
        kern = K(trainData, gamma, opts='test', testData = testData[testIdx,:])
        pred = np.sum(alpha*trainLabels*kern) + b
        if pred > 0:
            output.append(1)
        else:
            output.append(-1)

    output = np.asarray(output)
    acc = np.sum(output == testLabels)/output.size
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
    # calculate value of kernel
    kernVal = linKern(trainData)
    for k in range(len(C)):
        # train SVM with linear kernel
        alpha, b = trainKernSVM(trainData, trainLabels, kernVal, C[k]) 
        # calculate weight vector
        alpha_y = alpha*trainLabels
        w = np.zeros(trainData.shape[1])
        for alIdx in range(alpha.size):
            w = np.sum((alpha[alIdx]*trainLabels[alIdx])*trainData[alIdx,:]) + w
        # calculate b
        svIdx = np.argwhere(alpha>0)
        b = []
        for bIdx in svIdx:
            curX = np.squeeze(trainData[bIdx,:])
            b.append(trainLabels[bIdx] - np.dot(w, curX)) 
        b = np.asarray(b)
        b = np.mean(b) # average over all support vectors

        w = np.append(w, b)

        # calculate training and testing errors
        trainAcc, output = forwardLinSVM(w, trainData, trainLabels)
        trainError = 1 - trainAcc
        testAcc, output = forwardLinSVM(w, testData, testLabels)
        testError = 1 - testAcc
        # print errors
        print("For a C value of {} ".format(C[k]))
        print("Training Error: {}".format(trainError))
        print("Testing Error: {}".format(testError))
        print("\n")

        ben = 1




    #### Problem 3B ####
    gamma = [0.1, 0.5, 1, 5, 100]
    for gamVal in gamma:
        gaussK = gaussKern(trainData, gamVal)
        for k in range(len(C)):
            # train SVM with linear kernel
            alpha, b = trainKernSVM(trainData, trainLabels, gaussK, C[k]) 
            # forward pass on train data
            trainAcc, out = forwardKernSVM(alpha, b, gaussKern, gamVal, trainData, trainLabels, trainData, trainLabels)
            trainEr = 1 - trainAcc
            # forward pass on test data
            testAcc, out = forwardKernSVM(alpha, b, gaussKern, gamVal, testData, testLabels, trainData, trainLabels)
            testEr = 1 - testAcc
            print("For a C value of {} and gamma of {}".format(C[k], gamVal))
            print("Training Error: {}".format(trainEr))
            print("Testing Error: {}".format(testEr))
            print("\n")