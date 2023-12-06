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
    sig = 1/(1 + np.exp(-1*x))
    return sig


def tstWeights():
    w1 = np.asarray([[-1, -2, -3], [1, 2, 3]])
    w2 = np.asarray([[-1, -2, -3], [1, 2, 3]])
    w3 = np.asarray([-1, 2, -1.5])
    w3 = np.expand_dims(w3, axis=1)
    w = [w1, w2, w3]
    return w



# function to generate random weights from gaussian distribution
def genRandWeight(width,inSize):
    # weights to create the first hidden layer
    w1 = np.random.normal(size=(width-1,inSize+1)) # subtract 1 to account for bias node
    # weights to create the second hidden layer
    w2 = np.random.normal(size=(width-1, width))
    # weights to create the third hidden layer
    w3 = np.random.normal(size=(width, 1))
    # concatinate into a list
    w = [w1, w2, w3]
    return w

# function to perform forward pass through our 3 layer network
def NN3LayerForward(w, input):
    # check to see if we have a 1D or 2D array
    if np.size(input.shape) < 2:
        input = np.expand_dims(input, axis=0)
    out = np.zeros(input.shape[0])
    for inIdx in range(input.shape[0]):
        curIn = np.append([1], input[inIdx,:])
        # dot product with inputs and first layer of weights, then pass to sigmoid activation
        neuron1 = sigmoidActivation(np.dot(w[0], curIn))
        neuron1 = np.append([1], neuron1) # append with 1 to account for bias node
        # repeat for layer 2
        neuron2 = sigmoidActivation(np.dot(w[1], neuron1))
        neuron2 = np.append([1], neuron2)
        # layer 3
        neuron3 = np.dot(np.squeeze(w[2]), neuron2)
        NeuronVal = [neuron1, neuron2]
        out[inIdx] = neuron3
    # return output value and value of hidden layers, which is necessary for backpropagation
    return out, NeuronVal
        
# function to train our 3 layer NN using the backpropagation algorithm
def train3LayerNN(w_0, trainData, trainLabels, numEpoch):
    w = w_0.copy()
    y_0 = 1
    d = 3
    count = 0
    for epoch in range(numEpoch):
        # randomly shuffle the data
        shfIdx = list(range(trainData.shape[0]))
        random.shuffle(shfIdx)
        shfData = trainData[shfIdx,:]
        shfLabels = trainLabels[shfIdx]
        for dataIdx in range(trainData.shape[0]):
            curData = shfData[dataIdx,:]
            curLabel = shfLabels[dataIdx]
            gradients = backpropagation3LNN(w, curData, curLabel)
            count += 1
            lr = y_0/(1 + (y_0/d)*count)
            for wIdx in range(len(w)):
                w[wIdx] = w[wIdx] + lr*gradients[wIdx]

    return w

# function to perform backpropagation and calculate gradients
def backpropagation3LNN(w, data, label):
    # run forward pass on data
    prediction, nodes = NN3LayerForward(w, data)
    apData = np.append([1], data)
    dLdy = prediction[0] - label # since prediction is a numpy array
    # gradient of weights in layer 3 = dL/dy * dy/dw = (y - y^*) * (node associated with weight)
    grad3 = dLdy*nodes[1]
    grad3 = np.expand_dims(grad3, axis=1)
    # gradient of weights in layer 2
    dydnode2 = w[2][1:np.size(w[2])] 
    dnode2 = nodes[1][1:np.size(nodes[1])]*(1-nodes[1][1:np.size(nodes[1])]) # gradient of nodes in layer 2, except for bias
    dnode2 = np.expand_dims(dnode2, axis=1)
    grad2 = dLdy*dydnode2*dnode2*np.asarray([nodes[0], nodes[0]])
    # gradient of weights in layer 1
    dnode1 = nodes[0][1:np.size(nodes[0])]*(1-nodes[0][1:np.size(nodes[0])]) # gradient of nodes in layer 2, except for bias
    dnode1 = np.expand_dims(dnode1, axis=1)
    grad1 = np.zeros(w[0].shape)
    for nodIdx in range(dydnode2.size):
        dnode1dw = w[1][nodIdx,1:np.shape(w[1])[1]]
        dnode1dw = np.expand_dims(dnode1dw, axis=1)
        grad1 = dLdy*dydnode2[nodIdx,0]*dnode2[nodIdx,0]*dnode1dw*dnode1*np.asarray([apData, apData]) + grad1
    grads = [grad1, grad2, grad3]
    return grads




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
    w_0 = genRandWeight(3, trainData.shape[1])
    w = train3LayerNN(w_0, trainData, trainLabels, 2)
    

    ben = 1

    #### Problem 2C ####


