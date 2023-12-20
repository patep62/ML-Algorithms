import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import math

np.random.seed(1342)
sc = StandardScaler()

filePath = "C:\\Users\\ppart\\OneDrive\\Desktop\\School Stuff\\4SL4\\Assignment4\\data_banknote_authentication.txt"
raw_df = pd.read_csv(filePath, sep=",", header=None).to_numpy()
tMatrix = raw_df[:,-1]
XMatrix = np.delete(raw_df, -1, 1)

#First Splt 80/20
trainingSet_x, testSet_x, trainingSet_t, testSet_t = train_test_split(XMatrix,tMatrix,test_size=0.2,random_state=1342)
#Second Split, 0.25*0.8 = 0.2 -> 60/20/20
trainingSet_x, validationSet_x, trainingSet_t, validationSet_t = train_test_split(trainingSet_x,trainingSet_t,test_size=0.5,random_state=1342)

trainingSet_x = sc.fit_transform(trainingSet_x)
testSet_x = sc.transform(testSet_x)
validationSet_x = sc.transform(validationSet_x)

#Create sigmoid function to easily apply later.
def sigmoid(x):
    return 1/(1+ np.exp(-x))

#Write derivative of sigmoid to use during back prop.
def dSigmoid(x):
    return np.exp(-x)/(1+ np.exp(-x))**2

#Cross entropy loss function
def crossEntropyLoss(z, t):
    A = t * np.logaddexp(0,-z)
    B = (1-t) * np.logaddexp(0,z)

    return A + B

#Evaluate our prediction using a classifier.
def classifier(output):
    if output[0][0] > 0.5:
        return 1
    else:
        return 0

#Forward Propogation, using the sigmoid function as the activation for all stages.
def forwardPropogation(W1, W2, W3, X):

    #Hidden Layer 1
    z1 = np.dot(W1, np.vstack(([1], X)))
    h1 = sigmoid(z1)

    #Hidden Layer 2
    z2 = np.dot(W2, np.vstack(([1], h1)))
    h2 = sigmoid(z2)

    #Output Layer
    z3 = np.dot(W3, np.vstack(([1], h2)))
    output = sigmoid(z3)

    return output, z3, h2, z2, h1, z1

def backwardPropogation(z3, h2, z2, h1, z1, W2, W3, X, t):

    #Layer 3
    dz3 = -t + sigmoid(z3)
    h2 = np.hstack(([[1]],h2.T))
    dw3 = np.dot(dz3, (h2))
    W3 = np.delete(W3, 0, 1)
    dz2 = dSigmoid(z2) * np.dot(W3.T, dz3)

    #Layer 2
    h1 = np.hstack(([[1]],h1.T))
    dw2 = np.dot(dz2, (h1))
    W2 = np.delete(W2, 0, 1)
    dz1 = dSigmoid(z1) * np.dot(W2.T, dz2)

    #Layer 1
    X = np.hstack(([1], X))
    #print(dz1)
    dw1 = np.dot(dz1, (X.reshape(1,-1)))

    return dw1,dw2,dw3

def stochasticGradientDescent(X, t, validX, validt, n1, n2, numEpochs, alpha, inputSeed=1342):

    #Include a parameter to change to seed which will change the weight initialization
    np.random.seed(inputSeed)

    n = X.shape[0] #Num Examples
    m = X.shape[1] #Num Features

    #Shuffle examples in X and T matrices.
    indices = np.arange(n)
    np.random.shuffle(indices)
    X = X[indices]
    t = t[indices]

    #Initialize our vector of weights. The dimensions are determined by the size of the input layer and the size of the output layer.
    #.rand returns number between 0 to 1, I subtract 0.5 to get a number between -0.5 and 0.5.
    W1 = np.random.rand(n1,m) - 0.5 
    W2 = np.random.rand(n2,n1) - 0.5 
    W3 = np.random.rand(1,n2) - 0.5 #Weights for the output, only need 1 for binary classification.

    #Add column of ones for bias.
    W1 = np.hstack((np.ones(n1).reshape(-1,1), W1))
    W2 = np.hstack((np.ones(n2).reshape(-1,1), W2))
    W3 = np.hstack((np.ones(1).reshape(-1,1), W3))

    lowestValidError = 10000
    errorTrain = []
    errorValid = []
    epochs = []

    #Start the training
    for j in range(numEpochs):

        avgError = 0
        missClassed = 0
        
        for i in range(n): #Look through each example in our shuffled set.

            example = X[i] #Extract our example
            output, z3, h2, z2, h1, z1 = forwardPropogation(W1, W2, W3, example.reshape(-1,1)) #Foward Prop
            prediction = classifier(output) #Evaluate the prediction
            if prediction != t[i]: #Check if we're right
                missClassed += 1
            avgError += crossEntropyLoss(z3[0][0],t[i]) #Compute error
            dw1,dw2,dw3 = backwardPropogation(z3, h2, z2, h1, z1, W2, W3, example, t[i]) #Compute new updates
            
            #Update our parameters
            W1 = W1 - alpha * dw1
            W2 = W2 - alpha * dw2
            W3 = W3 - alpha * dw3
        
        #Compute average error and missclassification rate
        avgError /= n 
        missClassed /= n

        #Save train/valid errors for plotting.
        errorTrain.append(avgError)

        #Compute error on the validation set using our newly derived weights.
        validError, MR = getPredictionErrorNN(W1,W2,W3,validX,validt)
        errorValid.append(validError)
        epochs.append(j)
        
        #Output results every 50 epochs, also check validation error to stop early.
        if(j%20 == 0 or j == numEpochs-1):
            #print("Epoch: " + str(j) + " Cross Entropy Loss: " + str(validError) + " Missclassification Rate: " + str(missClassed))
            #If our validation error does not improve, break out and stop early
            if validError >= lowestValidError:
                break
            else:
                lowestValidError = validError
        
    testError, testMR = getPredictionErrorNN(W1,W2,W3,testSet_x,testSet_t)
    validationError, validMR = getPredictionErrorNN(W1,W2,W3,validX,validt)
    trainError, trainMR = getPredictionErrorNN(W1,W2,W3,X,t)

    print("Final Test Error = " + str(testError) + " Missclassification Rate = " + str(testMR))
    print("Final Validation Error = " + str(validationError) + " Missclassification Rate = " + str(validMR))
    print("Final Training Error = " + str(trainError) + " Missclassification Rate = " + str(trainMR))

    return errorTrain,errorValid,epochs,validationError

def getPredictionErrorNN(W1,W2,W3,Xtrain,tTrain):
    avgError = 0
    missClassed = 0
    n = Xtrain.shape[0]
    for i in range(n):
        example = Xtrain[i]
        output, z3, h2, z2, h1, z1 = forwardPropogation(W1, W2, W3, example.reshape(-1,1)) #Foward Prop
        prediction = classifier(output) #Evaluate the prediction
        if prediction != tTrain[i]: #Check if we're right
            missClassed += 1
        avgError += crossEntropyLoss(z3[0][0],tTrain[i]) #Compute error
    
    return avgError / n, missClassed / n

#Find best pair of n1/n2
def findBestN1N2(X, t, validX, validt):
    lowestValidError = 100000
    lowestValidSeed = 100000
    lowestTrainError = 10000
    bestN1 = 0
    bestN2 = 0
    seedList = [1342, 4132, 2134] #List of seeds to try different initial weights
    seed = 4132

    #Iterate through each possible n1, from (0+1) to (6+1) with n2 being 7-n1.
    for n1 in range(1,8):
        lowestValidSeed = 100000 #For each pair, try 3 different seeds or 3 different inital weights. Pick the lowest valid error from them.
        n2 = 8-n1
        print("N1 = " + str(n1) + " N2 = " + str((n2)))
        errorTrain,errorValid,epochs,validError = stochasticGradientDescent(X, t, validX, validt, n1, n2, 500, 0.005, seed)
        trainError = sum(errorTrain) / len(errorTrain)
        #print("N1 = " + str(n1) + " N2 = " + str((n2)) + " Validation Error = " + str(validError) + " Training Error = " + str(trainError))
        if validError < lowestValidError:
            lowestValidError = validError
            lowestTrainError = trainError
            bestN1 = n1
            bestN2 = n2
        
    print("Pair of N1 and N2 with the lowest validation error: " + str(bestN1) + "," + str(bestN2))

def plotLearningCurves(X, t, validX, validt):
    errorTrain,errorValid,epochs,lowestValidError = stochasticGradientDescent(X, t, validX, validt, 6, 2, 500, 0.005)
    plt.plot(epochs, errorTrain)
    plt.plot(epochs, errorValid)
    plt.title("Training and Validation Learing Curves")
    plt.xlabel("Time (Epochs)")
    plt.ylabel("Cross Entropy Loss")
    plt.legend(["Training Error", "Validation Error"])
    plt.show()

#errorTrain,errorValid,epochs,lowestValidError = stochasticGradientDescent(trainingSet_x, trainingSet_t, validationSet_x, validationSet_t, 6, 2, 250, 0.005)
plotLearningCurves(trainingSet_x, trainingSet_t, validationSet_x, validationSet_t)
findBestN1N2(trainingSet_x, trainingSet_t, validationSet_x, validationSet_t)