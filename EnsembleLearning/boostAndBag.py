import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import random as rand

# now import functions from custom decision tree implementation
# we assume our function is called from the CS6350 folder, which should occur if we run from the run script
sys.path.insert(0, os.getcwd() + "/DecisionTree") # add decision tree folder to system path
import DT_Practice as DT


# function to calculate weighted Information Gain
def WeightInfoGain(data, weights):
    dataShape = data.shape
    # start by calculating overall entropy
    labels = np.unique(data[:,dataShape[1]-1]) # find unique labels
    frac_label = np.empty((labels.size))
    for labelIdx in range(len(labels)): # iterate through each output label in dataset
        weightLocs = np.argwhere(data[:,dataShape[1]-1] == labels[labelIdx]) # get locations of specific labels
        frac_label[labelIdx] = np.sum(weights[weightLocs]) # sum weights of that label
    H_S = np.sum(-1*(frac_label)*np.log2(frac_label)) # calculate overall entropy
    attGain = np.empty(dataShape[1] - 1) # array to write entropy for each attribute to

    for attIdx in range(dataShape[1] - 1): # iterate through columns of data array -- going through Attributes
        att = np.unique(data[:,attIdx]) # get unique attribute values
        AttFrac = np.empty(len(att)) # array to store fraction of each attribute value
        H_S_v = np.empty(len(att)) # array to store entropy of each attribute value, H(S_v)

        for attValIdx in range(len(att)): # iterate through specific values of a single attribute
            attVal_Loc = np.argwhere(data[:,attIdx] == att[attValIdx])
            attWeights = weights[attVal_Loc]
            AttFrac[attValIdx] = np.sum(attWeights) # sum weights to use for final info gain calc
            attLabel = np.ones((labels.size)) 

            for labelIdx in range(len(labels)): # iterate through labels for specific attribute value
                labAttLoc = np.argwhere(data[attVal_Loc,dataShape[1]-1] == labels[labelIdx])
                attLabel[labelIdx] = np.sum(attWeights[labAttLoc[:,0]])
            # calculate entropy for attribute subset
            zeroIdx = np.argwhere(attLabel == 0) # need to adjust for 0 entries to aviod -inf*0 = nan
            if np.size(zeroIdx) > 0:
                attLabel[attLabel == 0] = 1
                H_S_v[attValIdx] = np.sum(-1*(attLabel/AttFrac[attValIdx])*np.log2(attLabel/AttFrac[attValIdx]))
            else: 
                H_S_v[attValIdx] = np.sum(-1*(attLabel/AttFrac[attValIdx])*np.log2(attLabel/AttFrac[attValIdx]))
        # now calculat information gain for each attribute
        attGain[attIdx] = H_S - np.sum(AttFrac*H_S_v)
    return attGain

# function to generate a tree with depth of 2 based on weighted info gain of training data
def stump(data, Attributes, AttributeVals, AttIdx, weights, gainFunction=WeightInfoGain): 
    #Note: We assume the data is a numpy array of strings, with the corresponding label in the final column
    # default gain function is information gain, but the user can specify other gain functions

    labels = data[:,data.shape[1]-1] # get labels from data
    # check if all labels are the same
    labCheck = np.sum(labels == labels[0])
    if labCheck == labels.size:
        return DT.tree(labels[0]) # returns leaf node, which is a tree structure with only a name
    else:
        gains = gainFunction(data, weights) # calculate information gain using selected function -- must accept data and weights
        # get attribute with maximum information gain
        att = np.argmax(gains)
        trueAttIdx = AttIdx[att]
        # split up data based on attribute
        subTrees = []
        # find most common label for attribute output
        for attIdx in range(len(AttributeVals[att])):
            valIdx = np.argwhere(data[:,att] == AttributeVals[att][attIdx]) # return row indices 
            attWeights = weights[valIdx]
            # if row vector is empty, then we need to assign the value the most common label of whole dataset
            if valIdx.size == 0:
                posslabels = np.unique(labels)
                labCount = np.empty(posslabels.size)
                for labIdx in range(len(posslabels)):
                    weightIdxs = np.argwhere(labels[valIdx] == posslabels[labIdx])
                    labCount[labIdx] = np.sum(attWeights[weightIdxs[:,0]])
                subTrees.append(DT.tree(posslabels[np.argmax(labCount)])) # create leafnode with most common output of dataset
            else: # we find most common output to assign as leaf node
                posslabels = np.unique(labels)
                labCount = np.empty(posslabels.size)
                for labIdx in range(len(posslabels)):
                    weightIdxs = np.argwhere(labels[valIdx] == posslabels[labIdx])
                    labCount[labIdx] = np.sum(attWeights[weightIdxs[:,0]])
                subTrees.append(DT.tree(posslabels[np.argmax(labCount)])) # add to list of leafnodes
        # Create and return tree with depth of 2
        return DT.tree(Attributes[att], trueAttIdx, subTrees, AttributeVals[att])


# this function computes the weighted error of a classifier
def computeWeightError(classifier, weights, trainData):
    labels = trainData[:,trainData.shape[1]-1] # labels should be in last column of training data array
    # now we generate array of predictions from our weak classifier
    predictions = np.empty(trainData.shape[0],dtype=np.dtype('U100')) # create numpy array to store predictions
    for rowIdx in range(trainData.shape[0]):
        curData = trainData[rowIdx, 0:trainData.shape[1]-1] # grab current row of data without label
        output = classifier.forward(curData) # pass through classifier
        predictions[rowIdx] = output
    # generate vector of comparisons for labels and predictions
    compar = (predictions == labels)
    compar = compar.astype(int)
    compar[compar == 0] = -1
    compar = compar.astype(float)
    e_t = (1/2) - (1/2)*np.sum(weights*compar) # calculate weighted error
    return e_t, compar




# function to perform Adaboost algorithm on training data for decision trees
def adaBoostTrees(trainData, attributes, attVals, numIter):
    D_i = np.ones(trainData.shape[0])/trainData.shape[0] # initialize weights, D
    # initialize list of alpha values
    alphaVals = []
    # initialize list to store weak learners
    stumpList = []
    for itIdx in range(numIter): # iterate number of times selected by user
        # create weak learner
        newStump = stump(trainData, attributes, attVals, list(range(len(attributes))), D_i, gainFunction=WeightInfoGain)
        stumpList.append(newStump)
        # compute vote of weak classifier
        # compare is y_i * h_t(x_i)
        e_t, compare = computeWeightError(newStump, D_i, trainData)
        curAlpha = (1/2)*np.log((1-e_t)/e_t)
        alphaVals.append(curAlpha)
        # update weights
        D_i = D_i*np.exp(-1*curAlpha*compare)
        Z_t = np.sum(D_i) # calculate normalization constant
        D_i = D_i/Z_t # apply normalization constant to weights  
        

    # return list of trees and alpha values
    return stumpList, alphaVals


# forward pass through adaboost model
def adaboostForward(models, alphas, testData):
    # find labels from data and cast to -1 or 1 --> note that there can only be 2 output labels for this to work
    testLabels = testData[:,testData.shape[1]-1]
    labels = np.unique(testData[:,testData.shape[1]-1]) 
    if np.size(labels) > 2:
        raise Exception("More than 2 unique labels, choose a more appropriate algorithm")
    # iterate through models and perform forward pass on data
    predictions = []
    for rowIdx in range(testData.shape[0]):
        curData = testData[rowIdx,0:testData.shape[1]-1] # get current row of data
        # run forward pass through each model
        output = [mod.forward(curData) for mod in models]
        output = np.asarray(output)
        # convert output labels to -1 or 1
        h_t = (output == labels[0]) 
        h_t = h_t.astype(int)
        h_t[h_t == 0] = -1
        h_t = h_t.astype(float)
        # calculate final output
        H_final = np.sum(np.asarray(alphas)*h_t)
        # now apply sgn function using if statement
        if H_final < 0:
            predictions.append(labels[1]) 
        else:
            predictions.append(labels[0])

    accuracy = (np.sum(predictions == testLabels)/testLabels.size)
    return predictions, accuracy


# forward pass through adaboost model with precomputed outputs, this offers significant speedup
def adaboostForwardPreComp(alphas, testData, modelOutputs, numModels=0):
    # model outputs should be a list of lists containing the outputs for the model
    # numModels tells the number of models to use when computing the final hypothsis, if 0 all models given will be used
    if numModels == 0:
        labels = testData[:,testData.shape[1]-1]
        Ulabels = np.unique(testData[:,testData.shape[1]-1])
        if np.size(Ulabels) > 2:
            raise Exception("More than 2 unique labels, choose a more appropriate algorithm")
        # convert model outputs to numpy array
        modOut = np.asarray(modelOutputs)
        predictions = []
        # iterate through data, calculating outputs from ensemble
        for labIdx in range(labels.size):
            # find where model outputs match which label
            h_t = (modOut[:, labIdx] == Ulabels[0])
            h_t[h_t == 0] = -1
            h_t = h_t.astype(float)
            # calculate final output
            H_final = np.sum(np.asarray(alphas)*h_t)
            # now apply sgn function using if statement
            if H_final < 0:
                predictions.append(labels[1]) 
            else:
                predictions.append(labels[0])

        accuracy = (np.sum(predictions == labels)/labels.size)
        return predictions, accuracy
    else:
        labels = testData[:,testData.shape[1]-1]
        Ulabels = np.unique(testData[:,testData.shape[1]-1])
        if np.size(Ulabels) > 2:
            raise Exception("More than 2 unique labels, choose a more appropriate algorithm")
        # convert model outputs to numpy array
        modOut = np.asarray(modelOutputs)
        predictions = []
        # iterate through data, calculating outputs from ensemble
        for labIdx in range(labels.size):
            # find where model outputs match which label
            h_t = (modOut[0:numModels, labIdx] == Ulabels[0])
            h_t[h_t == 0] = -1
            h_t = h_t.astype(float)
            # calculate final output
            H_final = np.sum(np.asarray(alphas[0:numModels])*h_t)
            # now apply sgn function using if statement
            if H_final < 0:
                predictions.append(labels[1]) 
            else:
                predictions.append(labels[0])

        accuracy = (np.sum(predictions == labels)/labels.size)
        return predictions, accuracy



def baggedTrees(numTrees, mSamples, trainData, Attributes, AttributeVals):
    treeList = [] 
    for treeIdx in range(numTrees):
        sampleIdxs = []
        for k in range(mSamples):
            # draw mSamples uniformly with replacement
            sampleIdxs.append(rand.randint(0, trainData.shape[0]-1))

        # grab sample data from overall set of training data
        sampleData = trainData[sampleIdxs,:]
        # create new tree from sample data using ID3 algorithm
        newTree = DT.ID3(sampleData, Attributes, AttributeVals, list(range(len(Attributes)))) 
        treeList.append(newTree)

    return treeList

    
def bagForwardPreComp(modelOutput, Data, numModels=0):
    labels = Data[:, Data.shape[1]-1]
    Ulabels = np.unique(labels)
    modOut = np.asarray(modelOutput) # make list of lists into numpy array
    if numModels == 0:
        labels = Data[:, Data.shape[1]-1]
        Ulabels = np.unique(labels)
        # iterate through unique labels, finding how many are in model outputs
        labCount = np.zeros((Ulabels.size, labels.size))
        for uIdx in range(Ulabels.size):
            modMatch = (modOut == Ulabels[uIdx])
            labCount[uIdx, :] = np.sum(modMatch, axis=0)
        labIdx = np.argmax(labCount, axis=0)
        predictions = Ulabels[labIdx]
        acc = np.sum(predictions == labels)/labels.size
        return acc, predictions
    else:
        labels = Data[:, Data.shape[1]-1]
        Ulabels = np.unique(labels)
        # iterate through unique labels, finding how many are in model outputs
        labCount = np.zeros((Ulabels.size, labels.size))
        for uIdx in range(Ulabels.size):
            modMatch = (modOut[0:numModels, :] == Ulabels[uIdx])
            labCount[uIdx, :] = np.sum(modMatch, axis=0)
        labIdx = np.argmax(labCount, axis=0)
        predictions = Ulabels[labIdx]
        acc = np.sum(predictions == labels)/labels.size
        return acc, predictions





# make sure this is the main file being called, otherwise we just want to use the functions, not run this part
if __name__ == "__main__": 
    # import bank data
    trainData = DT.LoadData("Data/bank/train.csv")
    testData = DT.LoadData("Data/bank/test.csv")

    # here the attributes are not as easy to grab from the file (and are not all present), so we maunually create them
    BankAtts = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 
                'campaign', 'pdays', 'previous', 'poutcome']
    job = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
        "blue-collar","self-employed","retired","technician","services"]
    marital = ["married","divorced","single"]
    education = ["unknown","secondary","primary","tertiary"]
    default = ["yes","no"]
    housing = ["yes","no"]
    loan = ["yes","no"]
    contact = ["unknown","telephone","cellular"]
    month = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    poutcome = ["unknown","other","failure","success"]

    BankAttVals = [['False', 'True'], job, marital, education, default, ['False', 'True'], housing, loan, contact, ['False', 'True'], month, 
                ['False', 'True'], ['False', 'True'], ['False', 'True'], ['False', 'True'], poutcome]
    

     # now convert numeric attributes to binary
    trainData = DT.convertNumData(trainData, BankAtts)
    testData = DT.convertNumData(testData, BankAtts)

    # test adaboost
    # TennisPath = 'Data/TennisData.csv'
    # TennisData = DT.LoadData(TennisPath)
    # TennisAtts = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    # TennisAttVal = [['S', 'O', 'R'], ['H', 'M', 'C'], ['H', 'N', 'L'], ['S', 'W']]

    # gains = DT.InfoGain(TennisData)

    # stumps, alphas = adaBoostTrees(TennisData, TennisAtts, TennisAttVal, 3)

    # predict, acc = adaboostForward(stumps, alphas, TennisData)


    # Problem 2A
    P2A = False
    if P2A:
        # go through 500 iterations of Adaboost
        stumps, alphas = adaBoostTrees(trainData, BankAtts, BankAttVals, 500)

        stumpTrainEr = []
        trainOutputs = []

        stumpTestEr = []
        testOutputs = []

        # iterate through stumps, calculating each stump's error and outputs
        for stpIdx in range(len(stumps)):
            curStump = stumps[stpIdx]
            stumpTrainAcc, stumpTrainOut = DT.testTree(curStump, trainData)
            stumpTrainEr.append((100 - stumpTrainAcc)/100)
            trainOutputs.append(stumpTrainOut)

            stumpTestAcc, stumpTestOut = DT.testTree(curStump, testData)
            stumpTestEr.append((100 - stumpTestAcc)/100)
            testOutputs.append(stumpTestOut)

        stumpNums = list(range(1, 501))

        plt.scatter(stumpNums, stumpTrainEr,label='Train Error')
        plt.scatter(stumpNums, stumpTestEr, label='Test Error')
        plt.xlabel("Stump Index")
        plt.ylabel("Stump Error")
        plt.legend()
        plt.savefig("Figures/StumpAcc.png")
        plt.close()

        trainAcc = []
        testAcc = []
        for T in range(1,501): # iterate through, grabbing a different number of stumps each time to get accuracy
            print(T)
            # get accuracy on training data -- use precomputed outputs from stumps for final prediction calculation
            trainPredict, curTrainAcc = adaboostForwardPreComp(alphas, trainData, trainOutputs, numModels=T)
            trainAcc.append(1-curTrainAcc)
            print("Training Error: ")
            print(1-curTrainAcc)
            # get accuracy on testing data --> 1 - accuracy to get error
            testPredict, curTestAcc = adaboostForwardPreComp(alphas, testData, testOutputs, numModels=T)
            testAcc.append(1-curTestAcc)
            print("Test Error: ")
            print(1-curTestAcc)

        numT = list(range(1,501))
        plt.plot(numT, trainAcc, label='Train Error')
        plt.plot(numT, testAcc, label='Test Error')
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig("Figures/AdaboostAcc.png")
        plt.close


    P2B = False
    if P2B:
        # generate 500 bagged trees
        bags = baggedTrees(500, 15, trainData, BankAtts, BankAttVals)
        # precompute output for each bagged tree
        trainOutput = []
        testOutput = []
        for bagIdx in range(len(bags)):
            # training data
            acc, bagTrainOutput = DT.testTree(bags[bagIdx], trainData)
            trainOutput.append(bagTrainOutput)
            # testing data
            acc, bagTestOutput = DT.testTree(bags[bagIdx], testData)
            testOutput.append(bagTestOutput)


        trainError = []
        testError = []
        for T in range(1,501):
            print(T)
            # now we hand precomputed outputs to function to calculate final hypothesis by majority vote
            trainAcc, predictions = bagForwardPreComp(trainOutput, trainData, T)
            trainError.append(1 - trainAcc)

            testAcc, predictions = bagForwardPreComp(testOutput, testData, T)
            testError.append(1 - testAcc)

        
        numIt = list(range(1,501))
        plt.plot(numIt, trainError, label='Train Error')
        plt.plot(numIt, testError, label='Test Error')
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig("Figures/BagAcc.png")
        plt.close


    P2C = True
    if P2C:
        bag100Trees = []
        for k in range(100):
            # randomly sample training dataset, 1000 samples, without replacement
            sampIdx = rand.sample(list(range(trainData.shape[0])), 1000) 
            sampData = trainData[sampIdx, :]
            # train 500 trees on those samples
            bagTrees = baggedTrees(500, 5, sampData, BankAtts, BankAttVals) 
            bag100Trees.append(bagTrees) 

        # take 100 bags and compute predictions
        allPredict = []
        for bagIdx in range(100):
            acc, predictions = DT.testTree(bag100Trees[bagIdx][0], testData)
            allPredict.append(predictions)
        
        # make predictions into numpy array
        allPredict = np.array(allPredict)
        # now we convert predictions to numerical values
        numPred = (allPredict == 'no')
        numPred = numPred.astype(int)
        numPred[numPred == 0] = -1
        numPred = numPred.astype(float)

        testLabels = testData[:, testData.shape[1]-1]
        testVals = (testLabels == 'no')
        testVals = testVals.astype(int)
        testVals[testVals == 0] = -1
        testVals = testVals.astype(float)

        # find average of predictions
        avPred = np.sum(numPred, axis=0)/numPred.shape[0]
        # calculate bias
        biaSingTree = np.power(testVals - avPred, 2)
        # compute sample variance of all predictions
        mHat = np.sum(numPred)/numPred.size
        sampSTD = np.sum(np.power(numPred - mHat, 2)) * (1/(numPred.size - 1))
        sampVar = np.sqrt(sampSTD)

        













