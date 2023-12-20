import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import sklearn.metrics as scikitMetrics
from sklearn.linear_model import LogisticRegression, SGDClassifier

np.random.seed(1342)
sc = StandardScaler()

#x_train, x_test = load_breast_cancer(return_X_y=True)
X,t = load_breast_cancer(return_X_y=True)

#Split data 80/20 training/validation
trainingSet_x, validationSet_x, trainingSet_t, validationSet_t = train_test_split(X,t,test_size=0.2,random_state=10)

dummy_values = np.ones(len(trainingSet_x))
dummy_values_validation = np.ones(len(validationSet_x))

#Insert dummy column and standardize
trainingSet_x = np.insert(trainingSet_x, 0, dummy_values, axis=1)
trainingSet_x = sc.fit_transform(trainingSet_x)
validationSet_x = np.insert(validationSet_x, 0, dummy_values_validation, axis=1)
validationSet_x = sc.transform(validationSet_x)

#Create sigmoid function to easily apply later.
def sigma(z):
    return 1/(1+ np.exp(-z))

#Cross entropy loss function
def crossEntropyLoss(z, t):
    A = t * np.logaddexp(0,-z)
    B = (1-t) * np.logaddexp(0,z)

    return A + B

#Batch Gradient Descent
def logisticRegression_BGD(X_matrix, T_matrix, alpha, numIterations):

    M = X_matrix.shape[0] # Examples
    N = X_matrix.shape[1] # Features

    w_vector = np.zeros((N)) #Initialize w vector
    cost_list = []

    #Start training, and keep going according to numIterations.
    for i in range(numIterations):

        #Compute z and y matrices.
        z = np.dot(X_matrix,w_vector)
        y = sigma(z)

        #Get average loss over all examples.
        loss = (1/M) * np.sum(crossEntropyLoss(z,T_matrix))

        #Get new set of w parameters
        w_vector = w_vector - alpha * (1/M)* np.dot(X_matrix.T,(y - T_matrix))

        cost_list.append(loss)
    print(w_vector)
    
    return cost_list, w_vector

#Stochastic Gradient Descent
def logisticRegression_SGD(X_matrix, T_matrix, alpha, numEpochs):

    N = X_matrix.shape[1] # Features
    M = X_matrix.shape[0] # Examples

    w_vector = np.zeros((N))
    cost_list = []

    #Shuffle examples in X and T matrices.
    indices = np.arange(M)
    np.random.shuffle(indices)
    X_matrix = X_matrix[indices]
    T_matrix = T_matrix[indices]

    #Outer loops for number of epochs
    for i in range(numEpochs):
        loss_list = []
        for j in range(M): #Inner loop iterating through each example.

            #Grab 1 example
            x_i = X_matrix[j]
            t_i = T_matrix[j]

            #Compute loss and new w parameters.
            z = np.dot(x_i,w_vector)
            y = sigma(z)

            loss = (1/M) * np.sum(crossEntropyLoss(z,t_i))
            loss_list.append(loss)
            w_vector = w_vector - alpha * x_i * (y - t_i)

        #Get average loss over entire epoch
        cost_list.append(np.sum(loss_list)/M)
    print(w_vector)
    
    return cost_list, w_vector

#plot learning curves
def learningCurvesBGD(trainingSet_x, trainingSet_t, alphaList, numIterations):

    legend = []
    x_axis = np.arange(0,numIterations,1)
    for alpha in alphaList:
        cost_list, w_vector = logisticRegression_BGD(trainingSet_x, trainingSet_t, alpha, numIterations)
        plt.plot(x_axis, cost_list)
        legend.append("alpha = " + str(alpha))
    
    plt.legend(legend)
    plt.title("Learning Rates for Batch Gradient Descent")
    plt.show()

#plot learning curves
def learningCurvesSGD(trainingSet_x, trainingSet_t, alphaList, numEpochs):

    legend = []
    x_axis = np.arange(0,numEpochs,1)
    for alpha in alphaList:
        cost_list, w_vector = logisticRegression_SGD(trainingSet_x, trainingSet_t, alpha, numEpochs)
        plt.plot(x_axis, cost_list)
        legend.append("alpha = " + str(alpha))
    
    plt.legend(legend)
    plt.title("Learning Rates for Stochastic Gradient Descent")
    plt.show()

#0 = Malignant (positive class), 1 = Benign (negative class)
def plotMetrics(w_vector, x, t):

    #Compute z
    z = np.dot(x, w_vector.T)

    #Initialize lists for plotting/analysis later
    missclassificationRateList = []
    precisionList = []
    recallList = []
    F1_scoreList = []
    fpRate_list = []
    tpRate_list = []

    #Get sorted z list to use as beta list
    beta_list = np.sort(z)

    #Outer loop iterating through each beta
    for beta in beta_list:

        #Initialize counters
        numMissclassified = 0
        FP = 0 #Total number of False Positives.
        TP = 0 #Total number of True Positives.
        P = 0 #Total number of positives. 
        N = 0 #Total number of negatives.

        #Inner loop iterating through each example.
        for i in range(len(t)):

            #Classify our predicted value into positive or negative class
            if z[i] >= beta:
                result = 1
            else:
                result = 0
            
            #Check if we are correct
            if result != t[i]:
                #We predicted incorrectly, increment missclassified total.
                numMissclassified += 1
                #Check if we predicted positive, if so, increment false positive total.
                if result == 0:
                    FP += 1
            #We predicted correctly.
            else:
                #Check if we predicted positive, if so, increment true positive total.
                if result == 0:
                    TP += 1
            
            #Increment positive or negative true totals.
            if t[i] == 0:
                P += 1
            else:
                N += 1
        
        #Edge cases to avoid division by zero
        if P == 0:
            recall = 0
        else:
            recall = TP / P
        if TP == 0 and FP == 0:
            precision = 0
            F1_score = 0
        else:
            precision = TP / (TP + FP)
            F1_score = 2 * precision * recall / (precision + recall)

        #Append to lists
        precisionList.append(precision)
        recallList.append(recall)
        F1_scoreList.append(F1_score)
        missclassificationRateList.append(numMissclassified/len(t))    

        fpRate_list.append(FP / N)
        tpRate_list.append(TP / P)    
    
    #Plot results.
    plt.plot(recallList, precisionList, 'o')
    plt.title("PR Curve for various Beta values")
    plt.show()

    plt.plot(fpRate_list, tpRate_list, 'o')
    plt.title("ROC Graph")
    plt.show()
            
    return missclassificationRateList, F1_scoreList

#Scikit implementation of BGD
def logisticRegressionBGD_scikit(trainingSet_x, validationSet_x, trainingSet_t, validationSet_t):
    
    #Create model and train data.
    model = LogisticRegression(solver='lbfgs', max_iter=200)
    model.fit(trainingSet_x, trainingSet_t)

    #Predict on validation/test set using trained model.
    yPredict = model.predict_proba(validationSet_x)[:, 1]
    y = model.predict(validationSet_x)

    #Print the w vector
    print(model.coef_)

    #Computer all required performance metrics, using built-in scikit methods.
    missclassificationRate = 1 - scikitMetrics.accuracy_score(validationSet_t, y)
    confusionMatrix = scikitMetrics.confusion_matrix(validationSet_t, y)
    F1_score = scikitMetrics.f1_score(validationSet_t, y)

    precision, recall, ignore = scikitMetrics.precision_recall_curve(validationSet_t, yPredict)
    fpRate, tpRate, ignore = scikitMetrics.roc_curve(validationSet_t, yPredict)

    plt.plot(recall, precision)
    plt.title("PR curve with scikit")
    plt.show()

    plt.plot(fpRate, tpRate)
    plt.title("ROC curve with scikit")
    plt.show()

    return missclassificationRate, F1_score

#Scikit implementation of SGD
def logisticRegressionSGD_scikit(trainingSet_x, validationSet_x, trainingSet_t, validationSet_t):
    model = SGDClassifier(loss='log_loss', max_iter=200)
    model.fit(trainingSet_x, trainingSet_t)

    yPredict = model.predict_proba(validationSet_x)[:, 1]
    y = model.predict(validationSet_x)

    print(model.coef_)
    
    missclassificationRate = 1 - scikitMetrics.accuracy_score(validationSet_t, y)
    confusionMatrix = scikitMetrics.confusion_matrix(validationSet_t, y)
    F1_score = scikitMetrics.f1_score(validationSet_t, y)

    precision, recall, ignore = scikitMetrics.precision_recall_curve(validationSet_t, yPredict)
    fpRate, tpRate, ignore = scikitMetrics.roc_curve(validationSet_t, yPredict)

    plt.plot(recall, precision)
    plt.title("PR curve with scikit")
    plt.show()

    plt.plot(fpRate, tpRate)
    plt.title("ROC curve with scikit")
    plt.show()

    return missclassificationRate, F1_score

#Train Model based on Batch Gradient Descent and plot PR and ROC Curves.
cost, w_vector = logisticRegression_BGD(trainingSet_x, trainingSet_t, 0.1, 200)
missclassificationRateList, F1_scoreList = plotMetrics(w_vector, validationSet_x, validationSet_t)
print("Lowest Missclassification rate and Highest F1 score for Batch Gradient Descent: " + str(min(missclassificationRateList)) + ", " + str(max(F1_scoreList)))

#Train Model based on Stochastic Gradient Descent and plot PR and ROC Curves.
cost, w_vector = logisticRegression_SGD(trainingSet_x, trainingSet_t, 0.1, 200)
missclassificationRateList, F1_scoreList = plotMetrics(w_vector, validationSet_x, validationSet_t)
print("Lowest Missclassification rate and Highest F1 score for Stochastic Gradient Descent: " + str(min(missclassificationRateList)) + ", " + str(max(F1_scoreList)))

#Train model and output plots using scikit libraries.
missclassificationRate, F1_score = logisticRegressionBGD_scikit(trainingSet_x, validationSet_x, trainingSet_t, validationSet_t)
print("Lowest Missclassification rate and Highest F1 score for Scikit's Batch Gradient Descent: " + str(missclassificationRate) + ", " + str(F1_score))

missclassificationRate, F1_score = logisticRegressionSGD_scikit(trainingSet_x, validationSet_x, trainingSet_t, validationSet_t)
print("Lowest Missclassification rate and Highest F1 score for Scikit's Stochastic Gradient Descent: " + str(missclassificationRate) + ", " + str(F1_score))

#Plot learning curves with 3 different learning rates.
learningCurvesBGD(trainingSet_x,trainingSet_t,[0.001, 0.01, 0.1], 200)
learningCurvesSGD(trainingSet_x,trainingSet_t,[0.001, 0.01, 0.1], 200)
