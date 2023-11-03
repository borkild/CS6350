To run batch gradient descent to train a linear classifier:

w, costs = batchGD(x, y, gradFunc = gradLMS, costFunc = LMS, r = 0.1, tol = 10**-6, maxIter = 10000)

x = x-training data
y = training labels
gradFunc = function to use to calculate gradient of cost function -- default is least means squares (LMS)
costFunc = function to use to calculate cost function -- deafault is LMS
r = learning rate -- default is 0.1
tol = tolerence, or amount of change in w from one iteration to the next necessary to continue -- default is 10^-6
maxIter = maximum number of iterations to let the algorithm run, this is here in case we make r too big and algorithm doesn't converge -- default is 10k

w = outputed weights
costs = value of cost function at each iteration


To run stochastic gradient descent:

w, costs = SGD(x, y, gradFunc = gradLMS, costFunc = LMS, r = 0.1, tol = 10**-6, maxIter = 10000)

x = x-training data
y = training labels
gradFunc = function to use to calculate gradient of cost function -- default is least means squares (LMS)
costFunc = function to use to calculate cost function -- deafault is LMS
r = learning rate -- default is 0.1
tol = tolerence, or amount of change in w from one iteration to the next necessary to continue -- default is 10^-6
maxIter = maximum number of iterations to let the algorithm run, this is here in case we make r too big and algorithm doesn't converge -- default is 10k

w = outputed weights
costs = value of cost function at each iteration
