import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import random as rand

# now import functions from custom decision tree implementation
# we assume our function is called from the CS6350 folder, which should occur if we run from the run script
sys.path.insert(0, os.getcwd() + "/DecisionTree") # add decision tree folder to system path
import DT_Practice as DT 

# compute least-means-squared gradient 
def gradLMS(x, y, w):
    # we expect x, y, and w to be numpy arrays
    # the columns of x should correspond to x_1, x_2, ... x_n
    # can use with multiple training examples (more than 1 row of x) or just 1 training example 
    if len(x.shape) < 2: # means x is just a single row
        # calculate gradient
        gradient = -1*((y - np.dot(w,x))*x)
    else:
        gradient = np.zeros(x.shape[1]) # initialize vector to hold gradients
        for exIdx in range(x.shape[0]): # iterate through training examples
            # update gradient
            gradient = -1*((y[exIdx] - np.dot(w,x[exIdx,:]))*x[exIdx,:]) + gradient
    # return gradient after going through all training examples
    return gradient

# function to calculate the least-mean-squares cost
def LMS(x, y, w):
    # we expect x, y, and w to be numpy arrays
    # the columns of x should correspond to x_1, x_2, ... x_n
    # can use with multiple training examples (more than 1 row of x) or just 1 training example 
    cost = 0
    for exIdx in range(x.shape[0]):
        cost = (y[exIdx] - np.dot(w, x[exIdx,:]))**2 + cost
    cost = 0.5*cost
    return cost



def batchGD(x, y, gradFunc = gradLMS, costFunc = LMS, r = 0.1, tol = 10**-6, maxIter = 10000):
    # initialize weights
    w = np.zeros(x.shape[1])
    # initialize tolerence and iteration checks
    conv = 1
    count = 0
    # initialize cost list
    costs = []
    while conv > tol:
        # set max number of iterations in case we cannot converge
        if count > maxIter:
            break
        # calculate gradients
        grads = gradFunc(x, y, w)
        # update weights
        wNew = w - r*grads
        # check for convergence
        conv = np.linalg.norm(wNew - w)
        # make w equal to wNew
        w = np.copy(wNew)
        # calculate cost
        newCost = costFunc(x, y, w)
        costs.append(newCost)
        # add to counter
        count += 1
    # return weight vector and costs over training
    return w, costs
        




def SGD(x, y, gradFunc = gradLMS, costFunc = LMS, r = 0.1, tol = 10**-6, maxIter = 10000):
    # initialize weights
    w = np.zeros(x.shape[1])
    # initialize tolerence and iteration checks
    conv = 1
    count = 0
    # initialize cost list
    costs = []
    for iter in range(maxIter):
        # set max number of iterations in case we cannot converge
        if count > maxIter:
            break
        for exIdx in range(x.shape[0]): # iterate through examples
            # calculate gradient from single example
            grads = gradFunc(x[exIdx,:], y[exIdx], w)
            # update weights from single example
            w = w - r*grads
            # calculate cost
            newCost = costFunc(x, y, w)
            costs.append(newCost)
    # return weight vector and costs over training
    return w, costs




if __name__ == "__main__":
    # load in training data
    trainDataLoc = "Data/concrete/train.csv"
    trainData = DT.LoadData(trainDataLoc) 
    # data is loaded in as strings, need to convert to numbers
    trainData = trainData.astype(float)
    trainX = trainData[:, 0:trainData.shape[1]-1]
    trainY = trainData[:, trainData.shape[1]-1]

    # load in testing data
    testDataLoc = "Data/concrete/test.csv"
    testData = DT.LoadData(testDataLoc)
    # data is loaded in as strings, need to convert to numbers
    testData = testData.astype(float)
    testX = testData[:, 0:testData.shape[1]-1]
    testY = trainData[:, trainData.shape[1]-1]

    # problem 3A
    P3A = True
    if P3A:
        print("running Problem 3A")
        # run batch gradient descent on train data
        wBatch, costs = batchGD(trainX, trainY, r=0.01)
        # use weights to get cost of testing data
        testCost = LMS(testX, testY, wBatch)
        print("Cost with batch gradient descent: ")
        print(testCost)
        print("Weights from batch gradient descent: ")
        print(wBatch)

        # plot cost over training iterations
        iter = list(range(len(costs)))
        plt.figure()
        plt.plot(iter, costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.savefig("Figures/BatchGDCost.png")
        plt.close


    # problem 3B
    P3B = True
    if P3B:
        print("running Problem 3B")
        # run batch gradient descent on train data
        wSGD, SGDcosts = SGD(trainX, trainY, r=0.01, maxIter = 3000)
        # use weights to get cost of testing data
        testCost = LMS(testX, testY, wSGD)
        print("Cost with Stochastic gradient descent: ")
        print(testCost)
        print("Weights from stochastic gradient descent: ")
        print(wSGD)

        # plot cost over training iterations
        iter = list(range(len(SGDcosts)))
        plt.figure()
        plt.plot(iter, SGDcosts)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.savefig("Figures/SGDCost.png")
        plt.close

    
    # problem 3C
    P3C = True
    if P3C:
        print("running Problem 3C")
        trainX = np.transpose(trainX)
        # solve analytical solution
        wTrue = np.matmul(np.linalg.inv(np.matmul(trainX, np.transpose(trainX))), np.matmul(trainX, trainY))

        print("The weight vector calculated with the analytical solution: ")
        print(wTrue)
        



    ben = 1
