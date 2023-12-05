import os
import sys
import numpy as np
import random



# import functions from previous assignment .py
sys.path.insert(0, os.getcwd() + "/DecisionTree") # add decision tree folder to system path
import DT_Practice as DT  

sys.path.insert(0, os.getcwd() + "/Perceptron") # add perceptron folder to system path
import Percep as per


def sigmoidActivation(x):
    sig = 1/(1 + np.exp(x))
    return sig


# function to generate random weights from gaussian distribution
def genRandWeight(width,inSize):
    # weights to create the first hidden layer
    w1 = np.random.normal(size=(width,inSize))
    # weights to create the second hidden layer
    w2 = np.random.normal(size=(width, width))
    # weights to create the third hidden layer
    w3 = np.random.normal(size=(width, 1))
    # concatinate into a list
    w = [w1, w2, w3]
    return w

# function to perform forward pass through our 3 layer network
def NN3LayerForward(w, input):
    out = np.zeros(input.shape[0])
    for inIdx in range(input.shape[0]):
        curIn = input[inIdx,:]
        # dot product with inputs and first layer of weights, then pass to sigmoid activation
        neuron1 = sigmoidActivation(np.dot(w[0], curIn))
        # repeat for layer 2
        neuron2 = sigmoidActivation(np.dot(w[1], neuron1))
        # layer 3
        neuron3 = np.dot(np.squeeze(w[2]), neuron2)
        NeuronVal = [neuron1, neuron2]
        out[inIdx] = neuron3
    # return output value and value of hidden layers, which is necessary for backpropagation
    return out, NeuronVal
        
# function to train our 3 layer NN using the backpropagation algorithm
def train3LayerNN(w_0, trainData, trainLabels):
    pass 

# function to perform backpropagation and calculate gradients
def backpropagation3LNN(w, data, label):
    # run forward pass on data
    prediction, nodes = NN3LayerForward(w, data)
    dLdy = prediction - label
    # gradient of weights in layer 3 = dL/dy * dy/dw = (y - y^*) * (node associated with weight)
    grad3 = dLdy*nodes[1]
    # gradient of weights in layer 2




if __name__ == "__main__":
    # load in training data
    trainPath = "Data/bank-note/train.csv"
    trainData = DT.LoadData(trainPath)
    # load in testing data
    testPath = "Data/bank-note/test.csv"
    testData = DT.LoadData(testPath)
    # convert to floating point numbers
    trainData = trainData.astype(float)
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

    #### Problem 2B ####
    w = genRandWeight(5, trainData.shape[1])
    output = NN3LayerForward(w, trainData)

    ben = 1

    #### Problem 2C ####


