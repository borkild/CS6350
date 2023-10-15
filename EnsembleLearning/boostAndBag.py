import numpy as np
import os
import sys

# now import functions from custom decision tree implementation
# we assume our function is called from the CS6350 folder, which should occur if we run from the run script
sys.path.insert(0, os.getcwd() + "/DecisionTree") # add decision tree folder to system path
import DT_Practice as DT


# function to calculate weighted Information Gain
def WeightInfoGain(data, weights):
    dataShape = data.shape
    # start by calculating overall entropy
    labels = np.unique(data[:,dataShape[1]-1]) # find unique labels
    num_label = np.empty((labels.size))
    for labelIdx in range(len(labels)): # find amount of each label
        num_label[labelIdx] = np.sum(data[:,dataShape[1]-1] == labels[labelIdx])
    H_S = np.sum(-1*(num_label/dataShape[0])*np.log2(num_label/dataShape[0])) # calculate overall entropy
    attGain = np.empty(dataShape[1] - 1) # array to write entropy for each attribute to

    for attIdx in range(dataShape[1] - 1): # iterate through columns of data array -- going through Attributes
        att = np.unique(data[:,attIdx]) # get unique attribute values
        numAttVal = np.empty(len(att)) # array to store number of each attribute value
        H_S_v = np.empty(len(att)) # array to store entropy of each attribute value, H(S_v)

        for attValIdx in range(len(att)): # iterate through specific values of a single attribute
            attVal_Loc = np.argwhere(data[:,attIdx] == att[attValIdx])
            numAttVal[attValIdx] = attVal_Loc.size
            attLabel = np.empty((labels.size)) 

            for labelIdx in range(len(labels)): # iterate through labels for specific attribute value
                attLabel[labelIdx] = np.sum(weights[data[attVal_Loc,dataShape[1]-1] == labels[labelIdx]])
            # calculate entropy for attribute subset
            zeroIdx = np.argwhere(attLabel == 0) # need to adjust for 0 entries to aviod -inf*0 = nan
            if np.any(zeroIdx):
                attLabel[attLabel == 0] = numAttVal[attValIdx]
                H_S_v[attValIdx] = np.sum(-1*(attLabel)*np.log2(attLabel))
            else: 
                H_S_v[attValIdx] = np.sum(-1*(attLabel)*np.log2(attLabel))
        # now calculat information gain for each attribute
        attGain[attIdx] = H_S - np.sum((numAttVal/len(data[:,attIdx]))*H_S_v)
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
            # if row vector is empty, then we need to assign the value the most common label of whole dataset
            if valIdx.size == 0:
                posslabels = np.unique(labels)
                labCount = np.empty(posslabels.size)
                for labIdx in range(len(posslabels)):
                    labCount[labIdx] = np.count_nonzero(labels == posslabels[labIdx])
                subTrees.append(DT.tree(posslabels[np.argmax(labCount)])) # create leafnode with most common output of dataset
            else: # we find most common output to assign as leaf node
                posslabels = np.unique(labels)
                labCount = np.empty(posslabels.size)
                for labIdx in range(len(posslabels)):
                    labCount[labIdx] = np.count_nonzero(labels[valIdx] == posslabels[labIdx])
                subTrees.append(DT.tree(posslabels[np.argmax(labCount)])) # add to list of leafnodes
        # Create and return tree with depth of 2
        return DT.tree(Attributes[att], trueAttIdx, subTrees, AttributeVals[att])




# function to perform Adaboost algorithm
def adaBoost(trainData, numIter, modelType):
    pass


# make sure this is the main file being called, otherwise we just want to use the functions, not run this part
if __name__ == "__main__": 
    # import bank data
    trainData = DT.LoadData("C:/Users/bigso/Box/CS6350/HW2/bank/train.csv")
    # now convert numeric attributes to binary


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

