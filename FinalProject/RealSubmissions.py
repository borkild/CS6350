import os
import csv
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
import torch
import torchvision
import torchinfo

# now import functions from custom decision tree implementation -- we only use this to load in the data
# we assume our function is called from the CS6350 folder, which should occur if we run from the run script
sys.path.insert(0, os.getcwd() + "/DecisionTree") # add decision tree folder to system path
import DT_Practice as DT


# function to write output to csv file
def writeOutput(fileName, data, fields):
    with open(fileName, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(data)


# convert categorical data to numerical
def catToNum(data, attributes, attVals):
    numData = np.zeros(data.shape) # array to hold numerical data
    for attIdx in range(len(attributes)): # iterate through attributes
        for valIdx in range(len(attVals[attIdx])): # iterate through possible values of each attribute
            curVal = attVals[attIdx][valIdx]
            idxLoc = np.argwhere(data[:,attIdx] == curVal)
            numData[idxLoc, attIdx] = valIdx 
            if curVal == '?':
                outstr = "{} has {} unknowns".format(attributes[attIdx], idxLoc.size)
                print(outstr)
    return numData


# define simple network architecture
class Layer3NN(torch.nn.Module):
    def __init__(self):
        super(Layer3NN, self).__init__()
        self.Layers = torch.nn.Sequential(
            torch.nn.Linear(14, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 15),
            torch.nn.Dropout(p=0.33),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.33),
            torch.nn.Linear(15, 15),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.33),
            torch.nn.Linear(15, 10),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.33),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.33),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.33),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.Layers(x)
        return out

# define pytorch class for dataset
class SalPredict(torch.utils.data.Dataset):
    def __init__(self, dataArray, dataLabels, split):
        self.split = split
        # make all data 32 bit float point
        self.data = np.float32(dataArray) # should be a numpy array
        self.labels = np.expand_dims(np.float32(dataLabels), axis=1) # should also be a numpy array

    def __len__(self):
        return self.labels.size
    
    def __getitem__(self, idx):
        inputData = torch.from_numpy(self.data[idx,:]) # convert to torch tensor here
        label = torch.tensor(self.labels[idx])
        return inputData, label


def init_Custom_Weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)



if __name__ == "__main__":
    trainPath = "FinalProject/data/train_final.csv"
    testPath = "FinalProject/data/test_final.csv"
    # load in data
    trainData = DT.LoadData(trainPath)
    testData = DT.LoadData(testPath)
    # create list of attributes
    attributes = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race',
                  'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'] 
    # create attribute values
    age = ['False', 'True']
    workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'] 
    fnlwgt = ['False', 'True'] 
    education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                 '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'] 
    education_num = ['False', 'True'] 
    marital_status = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                   'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
                   'Armed-Forces', '?']
    relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?']
    race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?']
    sex = ['Female', 'Male', '?'] 
    capital_gain = ['False', 'True'] 
    capital_loss = ['False', 'True'] 
    hours_per_week = ['False', 'True'] 
    native_country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 
                      'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 
                      'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
                        'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 
                        'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']

    attribVals = [age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain,
                  capital_loss, hours_per_week, native_country]


    # preprocess data

    # pull attributes out of training data
    att = trainData[0, :]
    trainData = np.delete(trainData, (0), axis=0)
    # pull labels from training data
    trainLabels = trainData[:, trainData.shape[1]-1]
    trainLabels = trainLabels.astype(int)
    trainData = np.delete(trainData, trainData.shape[1]-1, axis=1)
    # for now we just convert numeric labels to binary based on being above or below the median
    #trainData = DT.convertNumData(trainData, attributes)
    # now replace categorical attributes with numeric label
    print("Converting Training Data")
    trainDataNum = catToNum(trainData, attributes, attribVals)

    # repeat for test data
    #testData = DT.convertNumData(testData, attributes)
    # delete attributes
    testData = np.delete(testData, (0), axis = 0)
    # pull Ids from test data
    IDs = testData[:, 0]
    testData = np.delete(testData, (0), axis=1)
    # now convert test data to numeric values
    print("Converting Testing Data")
    testDataNum = catToNum(testData, attributes, attribVals)



    # split training into train and validation sets -- make validation set have even number of each label
    numVal0 = int(round(trainDataNum.shape[0]*0.05))
    numVal1 = int(round(trainDataNum.shape[0]*0.05)) # will use 10% of training data for validation set
    idx0 = np.argwhere(trainLabels == 0)
    idx1 = np.argwhere(trainLabels == 1)
    idxList0 = list(range(idx0.shape[0])) 
    idxList1 = list(range(idx1.shape[0]))
    valIdx0 = random.sample(idxList0, numVal0) # indices to use as validation data
    valIdx1 = random.sample(idxList1, numVal1)
    # write validation set to new array
    valIdx = np.append(idx0[valIdx0], idx1[valIdx1])
    valDataNum = trainDataNum[valIdx, :]
    valLabels = trainLabels[valIdx]
    # delete validation set from training data
    trainDataNum = np.delete(trainDataNum, valIdx, axis=0)
    trainLabels = np.delete(trainLabels, valIdx)


    outFields = ['ID', 'Prediction']

    plainDT = False
    if plainDT:
        # generate decision trees with varying depths
        maxDepth = 25
        bestValAcc = 0
        bestDepth = 0
        for depth in range(3, maxDepth):
            # generate decision tree using sklearn
            dTree = tree.DecisionTreeClassifier(max_depth=depth)
            dTree = dTree.fit(trainDataNum, trainLabels)
            # run tree on training data
            trainPredict = dTree.predict(trainDataNum)
            trainAcc = np.sum(trainPredict == trainLabels)/trainLabels.size
            print("For a depth of {}: ".format(depth))
            print("training accuracy: {}".format(trainAcc))
            # run on validation data
            valPredict = dTree.predict(valDataNum)
            valAcc = np.sum(valPredict == valLabels)/valLabels.size
            print("validation accuracy: {}".format(valAcc))

            # run tree on testing data
            testPredict = dTree.predict(testDataNum)
            # output testing predictions to csv file
            outFileName = "FinalProject/Submissions/DecisionTree/DecisionTree{}.csv".format(depth)
            final_output = np.zeros((testDataNum.shape[0], 2))
            final_output[:,0] = IDs
            final_output[:,1] = testPredict
            final_output = final_output.astype(int)
            writeOutput(outFileName, final_output, outFields)
            # write out depth that gave the maximum value
            if valAcc > bestValAcc:
                bestValAcc = valAcc
                bestDepth = depth

        print("A depth of {} had the best validation accuracy".format(bestDepth))



    AB = False
    if AB:
        valAcc = 0
        trainAccList = []
        valAccList = []
        for numStumps in range(2,250):
            # generate ensemble
            boost = ensemble.AdaBoostClassifier(n_estimators=numStumps)
            boost = boost.fit(trainDataNum, trainLabels)
            # run on training data
            trainPredict = boost.predict(trainDataNum)
            trainAcc = np.sum(trainPredict == trainLabels)/trainLabels.size
            trainAccList.append(trainAcc)
            print("For an ensemble of {} trees: ".format(numStumps))
            print("training accuracy: {}".format(trainAcc))
            # run on validation data
            valPredict = boost.predict(valDataNum)
            curValAcc = np.sum(valPredict == valLabels)/valLabels.size
            valAccList.append(curValAcc)
            print("validation accuracy: {}".format(curValAcc))
            if curValAcc > valAcc:
                print("New best accuracy! Running on test set")
                # run tree on testing data
                testPredict = boost.predict(testDataNum)
                valAcc = curValAcc
        # output testing predictions to csv file
        outFileName = "FinalProject/Submissions/AdaboostDT.csv"
        final_output = np.zeros((testDataNum.shape[0], 2))
        final_output[:,0] = IDs
        final_output[:,1] = testPredict
        final_output = final_output.astype(int)
        writeOutput(outFileName, final_output, outFields)

        # plot accuracies over number of trees
        numstump = list(range(2,250))
        plt.figure()
        plt.plot(numstump, trainAccList, label='Train Accuracy')
        plt.plot(numstump, valAccList, label='Val Accuracy')
        plt.xlabel("Number of Stumps")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("FinalProject/figures/BagAcc.png")
        plt.close

    tstSVM = True
    if tstSVM:
        kernList = ["linear", "poly", "rbf"]
        for k in range(len(kernList)):
            kern = kernList[k]
            # initialize SVM model with desired kernel
            SVMmod = svm.SVC(kernel=kern)
            # train SVM
            SVMmod.fit(trainDataNum, trainLabels)
            # get training accuracy
            trainPred = []
            for tIdx in range(trainDataNum.shape[0]):
                curData = trainDataNum[tIdx,:]
                curData = np.expand_dims(curData, axis=0)
                output = SVMmod.predict(curData)
                trainPred.append(output)
            trainPred = np.asarray(trainPred)
            trainAcc = np.sum(trainLabels == np.squeeze(trainPred))/np.size(trainPred)

            # test on validation set
            predictions = []
            for tstIdx in range(valDataNum.shape[0]):
                curData = valDataNum[tstIdx,:]
                curData = np.expand_dims(curData,axis=0)
                output = SVMmod.predict(curData)
                predictions.append(output)
            predictions = np.asarray(predictions)
            valAcc = np.sum(valLabels == np.squeeze(predictions))/np.size(predictions)
            print("kernel: " + kern)
            print("Training Accuracy: {}".format(trainAcc))
            print("Validation Accuracy: {}".format(valAcc))
            
            # run model on testing data
            tstPred = []
            for tstIdx in range(testDataNum.shape[0]):
                curData = testDataNum[tstIdx, :]
                curData = np.expand_dims(curData, axis=0)
                output = SVMmod.predict(curData)
                tstPred.append(output)

            tstPred = np.squeeze(np.asarray(tstPred))
            final_output = np.zeros((testDataNum.shape[0], 2))
            final_output[:,0] = IDs
            final_output[:,1] = tstPred
            final_output = final_output.astype(int)
            outFileName = "FinalProject/Submissions/SVM_" + kern + ".csv"
            writeOutput(outFileName, final_output, outFields) 


    RBFtry = True
    if RBFtry:
        # initialize SVM model with desired kernel
            SVMmod = svm.SVC(kernel="rbf", gamma='auto', C=10)
            # train SVM
            SVMmod.fit(trainDataNum, trainLabels)
            # get training accuracy
            trainPred = []
            for tIdx in range(trainDataNum.shape[0]):
                curData = trainDataNum[tIdx,:]
                curData = np.expand_dims(curData, axis=0)
                output = SVMmod.predict(curData)
                trainPred.append(output)
            trainPred = np.asarray(trainPred)
            trainAcc = np.sum(trainLabels == np.squeeze(trainPred))/np.size(trainPred)

            # test on validation set
            predictions = []
            for tstIdx in range(valDataNum.shape[0]):
                curData = valDataNum[tstIdx,:]
                curData = np.expand_dims(curData,axis=0)
                output = SVMmod.predict(curData)
                predictions.append(output)
            predictions = np.asarray(predictions)
            valAcc = np.sum(valLabels == np.squeeze(predictions))/np.size(predictions)
            print("kernel: RBF with Tuning")
            print("Training Accuracy: {}".format(trainAcc))
            print("Validation Accuracy: {}".format(valAcc))
            
            # run model on testing data
            tstPred = []
            for tstIdx in range(testDataNum.shape[0]):
                curData = testDataNum[tstIdx, :]
                curData = np.expand_dims(curData, axis=0)
                output = SVMmod.predict(curData)
                tstPred.append(output)

            tstPred = np.squeeze(np.asarray(tstPred))
            final_output = np.zeros((testDataNum.shape[0], 2))
            final_output[:,0] = IDs
            final_output[:,1] = tstPred
            final_output = final_output.astype(int)
            outFileName = "FinalProject/Submissions/SVM_paramRBF.csv"
            writeOutput(outFileName, final_output, outFields)


    
    tstNN = False
    if tstNN:
        print(np.sum(trainLabels))
        print(np.size(trainLabels))
        # check for gpu
        if torch.cuda.device_count() == 0: # if no GPU was found, throw error
            raise Exception("No GPU found")
        print(torch.cuda.get_device_name(0))
        # initialize pytorch dataloaders -- we use these because they take care of shuffling the data through each epoch for us
        trainDataset = SalPredict(trainDataNum, trainLabels, split='train')
        valDataset = SalPredict(valDataNum, valLabels, split='Validation')
        trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=trainDataNum.shape[0], shuffle=True)
        valDataLoader = torch.utils.data.DataLoader(valDataset, batch_size=valDataNum.shape[0], shuffle=True)
        # initialize model and send to gpu
        simpleNN = Layer3NN()
        simpleNN.apply(init_Custom_Weight)
        simpleNN.cuda()
        print(torchinfo.summary(simpleNN), (1, 14))
        
        # initialize optimizer
        optimizer = torch.optim.Adam(simpleNN.parameters(), lr = 0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        # initialize loss function
        loss = torch.nn.MSELoss()
        # intialize other training parameters and arrays to hold values
        numEpoch = 400
        trainAcc = []
        trainLoss = []
        valAcc = []
        valLoss = []
        best_Acc = 0
        for epoch in range(numEpoch): # iterate through epochs
            simpleNN.train()
            print("epoch " + str(epoch))
            for batchIdx, (data, label) in enumerate(trainDataLoader): # iterate through batches -- in this case the gpu is big enough to load all samples at once
                # zero the gradients
                optimizer.zero_grad()
                # send data and labels to gpu
                data = data.cuda()
                label = label.cuda()
                # run data through model
                output = simpleNN(data)
                # compute loss
                loss_val = loss(output, label)
                # calculate gradients with backpropagation
                loss_val.backward()
                # update parameters with optimizer
                optimizer.step() 
            # get training and validation accuracy 
            for batchIdx, (data,label) in enumerate(trainDataLoader):
                simpleNN.eval()
                with torch.no_grad():
                    data = data.cuda()
                    label = label.cuda()
                    predict = simpleNN(data)
                    lossV = loss(predict, label)
                    predict = predict.cpu().detach().numpy() # bring prediction back from gpu memory
                    label = label.cpu().detach().numpy()
                    trainLoss.append(lossV.item())
                    predict = np.float32(predict > 0.5)
                    acc = np.sum(predict == label)/np.size(label)
                    trainAcc.append(acc)
                    print("Training Accuracy: {}".format(acc))

            for batchIdx, (data,label) in enumerate(valDataLoader):
                simpleNN.eval()
                with torch.no_grad():
                    data = data.cuda()
                    label = label.cuda()
                    predict = simpleNN(data)
                    lossV = loss(predict, label)
                    predict = predict.cpu().detach().numpy() # bring prediction back from gpu memory
                    label = label.cpu().detach().numpy()
                    predict = np.float32(predict > 0.5)
                    acc = np.sum(predict == label)/np.size(label)
                    valLoss.append(lossV.item())
                    valAcc.append(acc)
                    print("Validation Accuracy: {}".format(acc))
                    if best_Acc < acc:
                        best_Acc = acc
                        tstData = torch.from_numpy(np.float32(testDataNum))
                        tstData = tstData.cuda()
                        tstPredict = simpleNN(tstData)
                        tstPredict = tstPredict.cpu().detach().numpy()
                        tstPredict = np.int32(tstPredict > 0.5)
                        final_output = np.zeros((testDataNum.shape[0], 2))
                        final_output[:,0] = IDs
                        final_output[:,1] = np.squeeze(tstPredict)
                        final_output = final_output.astype(int)
                        outFileName = "FinalProject/Submissions/NN_3.csv"
                        writeOutput(outFileName, final_output, outFields)
                        
            
        epochs = list(range(numEpoch))
        plt.plot(epochs, valLoss)
        plt.plot(epochs, trainLoss)
        plt.show()


    tstNNrs = True
    if tstNNrs:
        print(np.sum(trainLabels))
        print(np.size(trainLabels))
        # check for gpu
        if torch.cuda.device_count() == 0: # if no GPU was found, throw error
            raise Exception("No GPU found")
        print(torch.cuda.get_device_name(0))
        # initialize pytorch dataloaders -- we use these because they take care of shuffling the data through each epoch for us
        # find samples with <50k and >50k labels and split to form even classes
        over50Idx = np.argwhere(trainLabels == 1)
        under50Idx = np.argwhere(trainLabels == 0)
        # swap samples with income <50k to address class imbalance
        under50Samp = random.sample(list(range(under50Idx.size)), over50Idx.size)
        newTrainIdx = np.concatenate((over50Idx, under50Idx[under50Samp]))
        newtrainData = trainDataNum[newTrainIdx, :]
        newtrainLabels = trainLabels[newTrainIdx]

        trainDataset = SalPredict(newtrainData, newtrainLabels, split='train')
        trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=newtrainData.shape[0], shuffle=True)

        valDataset = SalPredict(valDataNum, valLabels, split='Validation')
        valDataLoader = torch.utils.data.DataLoader(valDataset, batch_size=valDataNum.shape[0], shuffle=True)
        # initialize model and send to gpu
        simpleNN = Layer3NN()
        simpleNN.apply(init_Custom_Weight)
        simpleNN.cuda()
        print(torchinfo.summary(simpleNN), (1, 14))
        
        # initialize optimizer
        optimizer = torch.optim.Adam(simpleNN.parameters(), lr = 0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        # initialize loss function
        loss = torch.nn.MSELoss()
        # intialize other training parameters and arrays to hold values
        numEpoch = 2000
        trainAcc = []
        trainLoss = []
        valAcc = []
        valLoss = []
        best_Acc = 0
        for epoch in range(numEpoch): # iterate through epochs
            # put network into training mode
            simpleNN.train()
            print("epoch " + str(epoch))
            for batchIdx, (data, label) in enumerate(trainDataLoader): # iterate through batches -- in this case the gpu is big enough to load all samples at once
                # zero the gradients
                optimizer.zero_grad()
                # send data and labels to gpu
                data = data.cuda()
                label = label.cuda()
                # run data through model
                output = simpleNN(data)
                # compute loss
                loss_val = loss(output, label)
                # calculate gradients with backpropagation
                loss_val.backward()
                # update parameters with optimizer
                optimizer.step() 
            # get training and validation accuracy 
            for batchIdx, (data,label) in enumerate(trainDataLoader):
                simpleNN.eval()
                with torch.no_grad():
                    data = data.cuda()
                    label = label.cuda()
                    predict = simpleNN(data)
                    lossV = loss(predict, label)
                    predict = predict.cpu().detach().numpy() # bring prediction back from gpu memory
                    label = label.cpu().detach().numpy()
                    trainLoss.append(lossV.item())
                    predict = np.float32(predict > 0.5)
                    acc = np.sum(predict == label)/np.size(label)
                    trainAcc.append(acc)
                    print("Training Accuracy: {}".format(acc))

            for batchIdx, (data,label) in enumerate(valDataLoader):
                simpleNN.eval()
                with torch.no_grad():
                    data = data.cuda()
                    label = label.cuda()
                    predict = simpleNN(data)
                    lossV = loss(predict, label)
                    predict = predict.cpu().detach().numpy() # bring prediction back from gpu memory
                    label = label.cpu().detach().numpy()
                    predict = np.float32(predict > 0.5)
                    acc = np.sum(predict == label)/np.size(label)
                    valLoss.append(lossV.item())
                    valAcc.append(acc)
                    print("Validation Accuracy: {}".format(acc))
                    if best_Acc < acc:
                        best_Acc = acc
                        tstData = torch.from_numpy(np.float32(testDataNum))
                        tstData = tstData.cuda()
                        tstPredict = simpleNN(tstData)
                        tstPredict = tstPredict.cpu().detach().numpy()
                        tstPredict = np.int32(tstPredict > 0.5)
                        final_output = np.zeros((testDataNum.shape[0], 2))
                        final_output[:,0] = IDs
                        final_output[:,1] = np.squeeze(tstPredict)
                        final_output = final_output.astype(int)
                        outFileName = "FinalProject/Submissions/NN_3.csv"
                        writeOutput(outFileName, final_output, outFields)
                        
            
        epochs = list(range(numEpoch))
        plt.figure()
        plt.plot(epochs, valLoss, label="Validation")
        plt.plot(epochs, trainLoss, label = "Training")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('FinalProject/figures/NNLoss.png')

        plt.figure()
        plt.plot(epochs, valAcc, label = 'Validation')
        plt.plot(epochs, trainAcc, label = "Training")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('FinalProject/figures/NNAcc.png')
        




