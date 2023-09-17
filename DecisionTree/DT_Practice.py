import numpy as np
import os
import math


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
                data[lineIdx,term_count] = val
                term_count += 1
    return data

# function to execute ID3 algorithm
def ID3(data,Attributes,max_depth): 
    #Note: We assume the data is a numpy array of strings, with the corresponding label in the final column
    pass 

# function to calculate Information Gain
def InfoGain(data):
    dataShape = data.shape
    # start by calculating overall entropy
    labels = np.unique(data[:,dataShape[1]-1]) # find unique labels
    print(labels.size)
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
                attLabel[labelIdx] = np.sum(data[attVal_Loc,dataShape[1]-1] == labels[labelIdx])
            # calculate entropy for attribute subset
            zeroIdx = np.argwhere(attLabel == 0) # need to adjust for 0 entries to aviod -inf*0 = nan
            if np.any(zeroIdx):
                attLabel[attLabel == 0] = numAttVal[attValIdx]
                H_S_v[attValIdx] = np.sum(-1*(attLabel/numAttVal[attValIdx])*np.log2(attLabel/numAttVal[attValIdx]))
            else: 
                H_S_v[attValIdx] = np.sum(-1*(attLabel/numAttVal[attValIdx])*np.log2(attLabel/numAttVal[attValIdx]))
        # now calculat information gain for each attribute
        attGain[attIdx] = H_S - np.sum((numAttVal/len(data[:,attIdx]))*H_S_v)
    return attGain


# function to calculate Majority Error
def ME(data):
    dataShape = data.shape
    # start by calculating overall entropy
    labels = np.unique(data[:,dataShape[1]-1]) # find unique labels
    print(labels.size)
    num_label = np.empty((labels.size))
    for labelIdx in range(len(labels)): # find amount of each label
        num_label[labelIdx] = np.sum(data[:,dataShape[1]-1] == labels[labelIdx])
    print(np.sum(num_label))
    print(num_label.max)
    ME_S = (np.sum(num_label) - np.max(num_label))/np.sum(num_label) # calculate overall entropy
    MEGain = np.empty(dataShape[1] - 1) # array to write ME for each attribute to

    for attIdx in range(dataShape[1] - 1): # iterate through columns of data array -- going through Attributes
        att = np.unique(data[:,attIdx]) # get unique attribute values
        numAttVal = np.empty(len(att)) # array to store number of each attribute value
        ME_S_v = np.empty(len(att)) # array to store ME of each attribute value, ME(S_v)

        for attValIdx in range(len(att)): # iterate through specific values of a single attribute
            attVal_Loc = np.argwhere(data[:,attIdx] == att[attValIdx])
            numAttVal[attValIdx] = attVal_Loc.size
            attLabel = np.empty((labels.size)) 

            for labelIdx in range(len(labels)): # iterate through labels for specific attribute value
                attLabel[labelIdx] = np.sum(data[attVal_Loc,dataShape[1]-1] == labels[labelIdx])
            # calculate entropy for attribute subset
            ME_S_v[attValIdx] = (numAttVal[attValIdx] - np.max(attLabel))/numAttVal[attValIdx]

        # now calculat information gain for each attribute
        MEGain[attIdx] = ME_S - np.sum((numAttVal/len(data[:,attIdx]))*ME_S_v) 
    return MEGain


# function to calculate Gini Index
def GI(data):
    dataShape = data.shape
    # start by calculating overall entropy
    labels = np.unique(data[:,dataShape[1]-1]) # find unique labels
    print(labels.size)
    num_label = np.empty((labels.size))
    for labelIdx in range(len(labels)): # find amount of each label
        num_label[labelIdx] = np.sum(data[:,dataShape[1]-1] == labels[labelIdx])
    GI_S = np.sum(-1*(num_label/dataShape[0])*np.log2(num_label/dataShape[0])) # calculate overall entropy
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
                attLabel[labelIdx] = np.sum(data[attVal_Loc,dataShape[1]-1] == labels[labelIdx])
            # calculate entropy for attribute subset
            zeroIdx = np.argwhere(attLabel == 0) # need to adjust for 0 entries to aviod -inf*0 = nan
            if np.any(zeroIdx):
                attLabel[attLabel == 0] = numAttVal[attValIdx]
                H_S_v[attValIdx] = np.sum(-1*(attLabel/numAttVal[attValIdx])*np.log2(attLabel/numAttVal[attValIdx]))
            else: 
                H_S_v[attValIdx] = np.sum(-1*(attLabel/numAttVal[attValIdx])*np.log2(attLabel/numAttVal[attValIdx]))
        # now calculat information gain for each attribute
        attGain[attIdx] = H_S - np.sum((numAttVal/len(data[:,attIdx]))*H_S_v)
    return attGain

# class to create our trees

# load in training data
CarTrainPath = 'data/car/train.csv' # assuming car data is in the same folder as script
CarTrainData = LoadData(CarTrainPath)

data = np.copy(CarTrainData)

print(CarTrainData[0,0])
H = InfoGain(data)
MEval = ME(data)
print(MEval)