import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor

np.random.seed(69)
sc = StandardScaler()

#Import Data set
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
t = raw_df.values[1::2, 2]

#Split data 80/20 training/validation
trainingSet_x, validationSet_x, trainingSet_t, validationSet_t = train_test_split(X,t,test_size=0.2,random_state=1342)

def nearestNeighbors(trainingSet_x, trainingSet_t, validationSet_x, validationSet_t, k):

    errorList = []

    for dataPoint in range(validationSet_x.shape[0]):

        distances = np.linalg.norm(trainingSet_x - validationSet_x[dataPoint], axis=1)

        #Since we need to match an x point to its target, we can't just sort normally.
        #We need to preserve the indexes of the distances since each one maps to a target in trainingSet_t.
        #Argsort will return the indexes of the values that would sort the array.
        shortestDistances = np.argsort(distances)

        #Now we can grab each t that corresponds to the k lowest distances, average them, and get our prediction.
        prediction = 0
        for i in range(k):
            prediction += trainingSet_t[shortestDistances[i]]
        prediction = prediction/k

        #print("Error = " + str(validationSet_t[dataPoint] - prediction))
        errorList.append((validationSet_t[dataPoint] - prediction)**2)
    
    #Return the list of MSEs
    return errorList
        
def crossValidation(trainingSet_x, trainingSet_t, numFolds, k):

    n = trainingSet_x.shape[0] #Number of examples
    foldLength = int(n / numFolds) #Number of examples per fold.

    validationErrorList = []
    trainingErrorList = []
    
    #Shuffle examples in X and T matrices.
    indices = np.arange(n)
    np.random.shuffle(indices)
    trainingSet_x = trainingSet_x[indices]
    trainingSet_t = trainingSet_t[indices]
    
    #Begin 5-Fold cross validation
    for i in range(numFolds):

        #Determine the indexes for the validation set for this fold.
        foldIndexStart = i*foldLength
        foldIndexEnd = (i+1)*foldLength
        
        #Get the new validation sets.
        newValidationSet_x = trainingSet_x[foldIndexStart : foldIndexEnd]
        newValidationSet_t = trainingSet_t[foldIndexStart : foldIndexEnd]

        #Edge case
        if i == 0:
            newTrainingSet_x = trainingSet_x[foldIndexEnd:n]
            newTrainingSet_t = trainingSet_t[foldIndexEnd:n]
        
        #Slice the training sets around the validation sets to get the remaining examples.
        else:
            newTrainingSet_x = np.concatenate((trainingSet_x[0:foldIndexStart], trainingSet_x[foldIndexEnd:n]))
            newTrainingSet_t = np.concatenate((trainingSet_t[0:foldIndexStart], trainingSet_t[foldIndexEnd:n]))

        #Perform k-NN on our validation set to get the cross-validation error.
        validationErrors = nearestNeighbors(newTrainingSet_x, newTrainingSet_t, newValidationSet_x, newValidationSet_t, k)

        #Perform k-NN on our training set to get the training error.
        trainingErrors = nearestNeighbors(newTrainingSet_x, newTrainingSet_t, newTrainingSet_x, newTrainingSet_t, k)

        #Append the average the MSEs for training and validation and append to list for plotting.
        validationErrorList.append(sum(validationErrors)/len(validationErrors))
        trainingErrorList.append(sum(trainingErrors)/len(trainingErrors))

    #Return the list of errors per fold.
    return trainingErrorList, validationErrorList

def plotCrossValidationError(trainingSet_x, trainingSet_t, kList):

    trainingErrorList = []
    validationErrorList = []

    #Get the training and validation errors for each k.
    for k in kList:
        trainingError,validationError  = crossValidation(trainingSet_x, trainingSet_t, 5, k)
        trainingErrorList.append(sum(trainingError)/len(trainingError))
        validationErrorList.append(sum(validationError)/len(validationError))

    #Plot the data.
    plt.plot(kList, trainingErrorList, 'o')
    plt.plot(kList, validationErrorList, 'o')
    plt.plot(kList, trainingErrorList)
    plt.plot(kList, validationErrorList)
    plt.title("Cross-Validation Error and Training Error vs k-NN")
    plt.xlabel("k")
    plt.ylabel("Training Error")
    plt.show()

    return validationErrorList.index(min(validationErrorList))+1, min(validationErrorList)

#Get training and validation errors for 0<k<81
kList = np.arange(1, 81, 1)
optimal_k, minError = plotCrossValidationError(trainingSet_x, trainingSet_t, kList)

print("Optimal k found = " + str(optimal_k) + " with MSE of " + str(minError))

testError = nearestNeighbors(trainingSet_x, trainingSet_t, validationSet_x, validationSet_t, optimal_k)
testError = sum(testError)/len(testError)
print("Test Error with optimal k: ", testError)