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
    def __init__(self, name = 'placehold', children = None, branches = None):
        self.name = str(name) # always convert name to a string, makes keeping track easier
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
        if len(self.branch) == 0 & len(self.child) == 0: # means we have hit a leaf node and need to return the value
            return self.name
        else:
            nodeList = np.arange(0, data.size) 
            nodeList = nodeList.astype(str)
            branchVal = data[nodeList == self.name]
            branchIdx = self.branch.index(branchVal[0])
            return self.child[branchIdx].forward(data)



# function to execute ID3 algorithm
def ID3(data, gainFunction=InfoGain,max_depth=0): 
    #Note: We assume the data is a numpy array of strings, with the corresponding label in the final column
    # default gain function is information gain, but the user can specify other gain functions
    # if max_depth is not set, then we will generate the tree until standard ID3 stop conditions are met

    labels = data[:,data.shape[1]-1] # get labels from data
    # check if all labels are the same
    labCheck = np.sum(labels == labels[0])
    if labCheck == labels.size:
        return tree(labels[0]) # returns leaf node, which is a tree structure with only a name
    else:
        gains = gainFunction(data) # calculate information gain using selected function
        # get attribute with maximum information gain
        att = np.argmax(gains)
        # find all possible values the attribute can take on
        possAttVal = np.unique(data[:,att]) 
        # split up data based on attribute
        







t = tree('1', [tree('2', [tree('4',[tree('5')])],['att3']), tree('3',tree('6'))], ['att1', 'att2'])

td = t.calcDepth()

print(td)

tnew = tree('2', [tree('1', [tree('-'), tree('+')], ['c','d']), 
                  tree('0', [tree('+'), tree('3', [tree('-'), tree('+')], ['g', 'h'])], ['a', 'b'])], ['e', 'f'])

example_data = np.array(['b', 'c', 'f', 'g'])
output = tnew.forward(example_data)
print(output)

# load in training data
CarTrainPath = 'DecisionTree/data/car/train.csv' # assuming car data is in the same folder as script
CarTrainData = LoadData(CarTrainPath)

data = np.copy(CarTrainData)

# pass training data through ID3 algorithm to generate tree
carTree = ID3(data)




print(CarTrainData[0,0])
H = InfoGain(data)
MEval = ME(data)
GIval = GI(data)
print(GIval)