To train Adaboost algorithm:

stumps, alphas = adaBoostTrees(trainData, Attributes, AttVals, nTrees)
Inputs:
trainData = training data, columns are attributes, rows are examples
attributes = list of attributes
AttVals = list of lists with possible values for each attribute
nTree = number of tree stumps to include in model
Outputs:
stumps: list of decision stumps from adaboost training
alphas: list of alpha values from adaboost training

To run data through adaboos model:

predictions, accuracy = adaboostForward(models, alphas, Data)

models = stumps outputted from adaBoost training
alphas = alpha values outputted from adaboost training
Data = data to pass through model, columns are attributes, rows are examples

predictions = list of model prediction, with each entry corresponding to each row in Data
accuracy = accuracy of the dataset


To train bagging algorithm:

treeList = baggedTrees(numTrees, mSamples, trainData, Attributes, AttributeVals)

numTrees = number of trees to include in model
mSamples = number of samples to pull from trainData to create each tree
trainData = training data, columns are attributes, rows are examples
Attributes = list of attributes
AttributeVals = list of lists with possible values for each attribute

treeList = list of trees output from the bagging algorithm


To train random forest:

treeList = RandomForestTrees(numTrees, mSamples, fAtts, trainData, Attributes, AttributeVals):

numTrees = number of trees to include in model
mSamples = number of samples to pull from trainData to create each tree
fAtts = number of attributes to pull from our data to construct each tree
trainData = training data, columns are attributes, rows are examples
Attributes = list of attributes
AttributeVals = list of lists with possible values for each attribute

treeList = list of trees output from the bagging algorithm
