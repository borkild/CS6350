import os
import csv
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble

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
    trainData = DT.convertNumData(trainData, attributes)
    # now replace categorical attributes with numeric label
    trainDataNum = catToNum(trainData, attributes, attribVals)

    # repeat for test data
    testData = DT.convertNumData(testData, attributes)
    # delete attributes
    testData = np.delete(testData, (0), axis = 0)
    # pull Ids from test data
    IDs = testData[:, 0]
    testData = np.delete(testData, (0), axis=1)
    # now convert test data to numeric values
    testDataNum = catToNum(testData, attributes, attribVals)



    # split training into train and validation sets
    numVal = int(round(trainDataNum.shape[0]*0.10)) # will use 10% of training data for validation set
    idxList = list(range(trainDataNum.shape[0])) 
    valIdx = random.sample(idxList, numVal) # indices to use as validation data
    # write validation set to new array
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


AB = True
if AB:
    valAcc = 0
    for numStumps in range(2,250):
        # generate ensemble
        boost = ensemble.AdaBoostClassifier(n_estimators=numStumps)
        boost = boost.fit(trainDataNum, trainLabels)
        # run on training data
        trainPredict = boost.predict(trainDataNum)
        trainAcc = np.sum(trainPredict == trainLabels)/trainLabels.size
        print("For an ensemble of {} trees: ".format(numStumps))
        print("training accuracy: {}".format(trainAcc))
        # run on validation data
        valPredict = boost.predict(valDataNum)
        curValAcc = np.sum(valPredict == valLabels)/valLabels.size
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







