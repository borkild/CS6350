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
                data[lineIdx,term_count] = val.strip('\n') # get rid of new line '\n'
                term_count += 1
    return data

# function to load in our attributes
def LoadAttribute(dataPath):
    with open(dataPath,'r') as f:
        dataArray = f.readlines() # read lines from file into list
        attributes = [] # empty list for attributes
        attVals = [] # empty list for values of attributes - this will actually be a list of lists
        for lineIdx in range(len(dataArray)): # iterate through list and break data up
            curline = dataArray[lineIdx]
            if 'label values' in curline: # get labels
                labels = dataArray[lineIdx+2].strip(' ').split(',')
                for labIdx in range(len(labels)):
                    labels[labIdx] = labels[labIdx].strip(' ')
                    labels[labIdx] = labels[labIdx].strip('\n')
            if 'attributes' in dataArray[lineIdx]: # get attributes and their values
                attEnd = lineIdx + 2
                attLine = ' '
                while attLine != '\n': # find where attributes end
                    attLine = dataArray[attEnd]
                    attEnd += 1

                attStart = lineIdx + 2
                attCount = 0
                attributes = [] # empty list for attributes
                attVals = [[] for k in range(attStart,attEnd-1)] # empty list for values of attributes - this will actually be a list of lists
                for attIdx in range(attStart,attEnd-1): # iterate through attributes
                    attLine = dataArray[attIdx]
                    initialSplit = attLine.split(' ') # split based on spaces to start
                    while (initialSplit.count('')): # get rid of blank entries if we need it
                        initialSplit.remove('') # get rid of blank entries
                    for splitIdx in range(len(initialSplit)): # process inidivual strings in list
                        initialSplit[splitIdx] = initialSplit[splitIdx].strip(',') # get rid of commas
                        initialSplit[splitIdx] = initialSplit[splitIdx].strip(':') # get rid of colon
                        initialSplit[splitIdx] = initialSplit[splitIdx].strip('.\n')
                    attributes.append(initialSplit[0]) # first entry in list should be attribute
                    attVals[attCount] = initialSplit[1:len(initialSplit)] # rest of entries are attribute values
                    attCount += 1
               
                
    return labels, attributes, attVals


# function to calculate Information Gain
def InfoGain(data):
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
    GI_S = 1 - np.sum(np.square(num_label/dataShape[0])) # calculate overall entropy
    attGain = np.empty(dataShape[1] - 1) # array to write entropy for each attribute to

    for attIdx in range(dataShape[1] - 1): # iterate through columns of data array -- going through Attributes
        att = np.unique(data[:,attIdx]) # get unique attribute values
        numAttVal = np.empty(len(att)) # array to store number of each attribute value
        GI_S_v = np.empty(len(att)) # array to store entropy of each attribute value, H(S_v)

        for attValIdx in range(len(att)): # iterate through specific values of a single attribute
            attVal_Loc = np.argwhere(data[:,attIdx] == att[attValIdx])
            numAttVal[attValIdx] = attVal_Loc.size
            attLabel = np.empty((labels.size)) 

            for labelIdx in range(len(labels)): # iterate through labels for specific attribute value
                attLabel[labelIdx] = np.sum(data[attVal_Loc,dataShape[1]-1] == labels[labelIdx])
            # calculate entropy for attribute subset
            GI_S_v[attValIdx] = 1 - np.sum(np.square(attLabel/numAttVal[attValIdx]))
        # now calculate Gini Index gain for each attribute
        attGain[attIdx] = GI_S - np.sum((numAttVal/len(data[:,attIdx]))*GI_S_v)
    return attGain

# class to create our trees
class tree:
    def __init__(self, name = 'placehold', idx=None, children = None, branches = None):
        self.name = str(name) # always convert name to a string, makes keeping track easier
        # column in which attribute lives in data -- used for forward pass through tree -- if = None means it is a leafnode
        self.index = idx 
        # branches are the values our attributes can take on
        if branches != None:
            if type(branches) != list:
                self.branch = [branches]
            else:
                self.branch = branches
        else:
            self.branch = []
        # children are the subtrees below our branches
        if children != None:
            if type(children) != list: # in case we add single child node and forget the brackets around it
                self.child = [children]
            else:
                self.child = children
        else:
            self.child = []
        # the indices of the branches match the indices of the subtree from that branch

    # function to add child trees and branches
    def addST(self, children=None, branches = None):
        self.child.append(children)
        self.branch.append(branches)

    # function to calculate depth of tree
    def calcDepth(self, depth = 0):
        curChild = self.child
        if len(curChild) == 0: # if there are no subtrees, then we are at the end of the tree
            depth += 1
            return depth
        else: # otherwise we need to go through the subtrees
            newdepth = depth
            for subtree in self.child:
                cur_stDepth = subtree.calcDepth(depth+1)
                if cur_stDepth > newdepth: # make sure we return the deepest part of the tree
                    newdepth = cur_stDepth
            return newdepth


    # function to do a forward pass through tree with data
    def forward(self,data):
        # note: data should be a row vector, with the names of the tree nodes corresponding to the indices of the attributes
        if self.index == None: # means we have hit a leaf node and need to return the value
            return self.name
        else:
            nodeList = np.arange(0, data.size) 
            branchVal = data[nodeList == self.index]
            branchIdx = self.branch.index(branchVal[0])
            return self.child[branchIdx].forward(data)


# function to handle testing of our trees, calculating the accuracy
def testTree(tree, testData):
    # we will iterate through the test data, passing it through the tree and checking the answer
    dataShape = testData.shape
    correctCount = 0
    for rowIdx in range(dataShape[0]-1): # iterate through the columns
        curData = testData[rowIdx,dataShape[1]-2] # get data without labels
        treeOut = tree.forward(curData) # pass current row through tree
        if treeOut == testData[rowIdx, dataShape[1]-1]:
            correctCount += 1 # if labels match, add 1 to count
    # calculate percent correct
    acc = (correctCount/dataShape[0])*100
    return acc


#### Problem 2a #####
# function to execute ID3 algorithm
def ID3(data, Attributes, AttributeVals, AttIdx, max_depth=0, gainFunction=InfoGain, count = 0): 
    #Note: We assume the data is a numpy array of strings, with the corresponding label in the final column
    # default gain function is information gain, but the user can specify other gain functions
    # if max_depth is not set, then we will generate the tree until standard ID3 stop conditions are met

    count += 1
    labels = data[:,data.shape[1]-1] # get labels from data
    # check if all labels are the same
    labCheck = np.sum(labels == labels[0])
    if labCheck == labels.size:
        return tree(labels[0]) # returns leaf node, which is a tree structure with only a name
    else:
        gains = gainFunction(data) # calculate information gain using selected function
        # get attribute with maximum information gain
        att = np.argmax(gains)
        # split up data based on attribute
        subTrees = []
        for attIdx in range(len(AttributeVals[att])):
            valIdx = np.argwhere(data[:,att] == AttributeVals[att][attIdx]) # return row indices 
            # if row vector is empty, then we need to assign the value the most common label
            if valIdx.size == 0:
                posslabels = np.unique(labels)
                labCount = np.empty(posslabels.size)
                for labIdx in range(len(posslabels)):
                    labCount[labIdx] = np.count_nonzero(labels == posslabels[labIdx])
                subTrees.append(tree(posslabels[np.argmax(labCount)])) # create leafnode
            else:
                subData = data[valIdx[:,0],:] # write subset of data to new array
                subData = np.delete(subData, att, 1)
                
                # check to make sure we still have attributes if we delete the one we just processed
                if len(Attributes) == 1:
                    # use most common label for final branch for each attribute
                    for attValIdx in range(len(AttributeVals[att])):
                        labelVals = labels[valIdx]
                        unqLab = np.unique(labelVals)
                        numLabel = np.empty(unqLab.size)
                        for labIdx in range(len(unqLab)):
                            numLabel[labIdx] = np.count_nonzero(labelVals == unqLab[labIdx])
                        newName = unqLab[np.argmax(numLabel)]
                        subTrees.append(tree(newName))
                else: # otherwise we will delete attribute before recursing
                    newAttributes = Attributes.copy()
                    newAttributeVals = AttributeVals.copy()
                    newAttIdx = AttIdx.copy()
                    trueAttIdx = newAttIdx[att]
                    # delete attribute we just used from lists
                    del newAttributes[att]
                    del newAttributeVals[att] 
                    del newAttIdx[att]

                # check to make sure tree isn't bigger than max depth -- use counter
                if (max_depth != 0) & (count == max_depth): 
                    # if tree is at the max_depth, then we need to return leaf node for the attribute values
                    posslabels = np.unique(labels)
                    labCount = np.empty(posslabels.size)
                    for labIdx in range(len(posslabels)):
                        labCount[labIdx] = np.count_nonzero(labels == posslabels[labIdx])
                    subTrees.append(tree(posslabels[np.argmax(labCount)])) # create leafnode of most common label
                else:
                    # repeat ID3
                    subTrees.append(ID3(subData, newAttributes, newAttributeVals, newAttIdx, max_depth, gainFunction, count))
        # Create and return tree
        return tree(Attributes[att], trueAttIdx, subTrees, AttributeVals[att])
        

# This was all testing stuff to validate functions

# tnew = tree('2', 2, [tree('1', 1, [tree('-'), tree('+')], ['c','d']), 
#                   tree('0', 0, [tree('+'), tree('3', 3, [tree('-'), tree('+')], ['g', 'h'])], ['a', 'b'])], ['e', 'f'])

# example_data = np.array(['b', 'c', 'f', 'g'])
# output = tnew.forward(example_data)
# print(output)

# TennisPath = 'data/TennisData.csv'
# TennisData = LoadData(TennisPath)
# TennisAtts = ['Outlook', 'Temperature', 'Humidity', 'Wind']
# TennisAttVal = [['S', 'O', 'R'], ['H', 'M', 'C'], ['H', 'N', 'L'], ['S', 'W']]

# #TennisTree = ID3(TennisData, TennisAtts, TennisAttVal, list(range(len(TennisAtts))))
# #TennisInput = np.array(['O', 'H', 'H', 'W'])
# #tenout = TennisTree.forward(TennisInput)
# #print(tenout)

# MEval = ME(TennisData)
# print(MEval)
# GIval = GI(TennisData)
# print(GIval)


###### Problem 2b #######
# load in training data
CarTrainPath = 'data/car/train.csv' # assuming car data is in the same folder as script
CarAttPath = 'data/car/data-desc.txt'
CarTrainData = LoadData(CarTrainPath)
CarLabels, CarAtts, CarAttVals = LoadAttribute(CarAttPath)
# load testing data
CarTestPath = 'data/car/test.csv'
CarTestData = LoadData(CarTestPath)


data = np.copy(CarTrainData)


# generate tree of depths 1-6 with information gain and run on test set
for treeDepth in range(1,6):
    carTree =  ID3(data, CarAtts, CarAttVals, list(range(len(CarAtts))), treeDepth)
    


# generate tree of depths 1-6 with majority error


# generate tree of depths 1-6 with gini index



