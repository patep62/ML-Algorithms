from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

#Initializing Parameters
sc = StandardScaler()
np.random.seed(1342)
variance = 0.0625
std_dev = np.sqrt(0.0625)
noise = np.random.normal(0, std_dev, 12).reshape(-1,1)
noise_validation = np.random.normal(0, std_dev, 120).reshape(-1,1)

#Initializing Training and Validation Sets
dummy_values = np.ones(12)
dummy_values_validation = np.ones(120)
trainingSet_x = np.arange(0, 1+1/11, 1/11).reshape(-1,1)
trainingSet_t = (np.sin((4 * np.pi * trainingSet_x) + (np.pi / 2)) + noise).reshape(-1,1)

validationSet_x = np.arange(0, 1+1/119, 1/119).reshape(-1,1)
validationSet_t = (np.sin((4 * np.pi * validationSet_x) + (np.pi / 2)) + noise_validation).reshape(-1,1)
true_t = (np.sin((4 * np.pi * validationSet_x) + (np.pi / 2))).reshape(-1,1)

#Initializing X matrix and validation matrix for computations with dummy values.
X_matrix = dummy_values.reshape(-1,1)
X_matrix_validation = dummy_values_validation.reshape(-1,1)

#Regularization Coefficient
lambda_coeff = 0.0013

#Lists to store training/validation errors for each M
training_error = np.zeros(12)
validation_error = np.zeros(12)

#Creating seperate matrixs with standardized values to be used for regularization
trainingSetStandardized_x = sc.fit_transform(trainingSet_x)
validationSetStandardized_x = sc.transform(validationSet_x)
X_matrixStandardized = X_matrix
X_matrixStandardized_validation = X_matrix_validation

#Loop through each M from 0 to 11
for i in range(12):
    #If M is 0, don't add new feature vector, compute with just dummy values.
    if i == 0:
        pass
    else:
        #Append new feature vector based on the training/validation matrix raised to the power of M
        X_matrix = np.append(X_matrix, np.power(trainingSet_x, i), axis=1)
        X_matrix_validation = np.append(X_matrix_validation, np.power(validationSet_x, i), axis=1)

        #Keep building standardized matrix for later on
        X_matrixStandardized = np.append(X_matrixStandardized, np.power(trainingSetStandardized_x, i), axis=1)
        X_matrixStandardized_validation = np.append(X_matrixStandardized_validation, np.power(validationSetStandardized_x, i), axis=1)

    #if M=11, we need to compute with regularization
    if i == 11:

        #Create B Matrix based on set lambda
        B = np.diag(np.full(12, 2*lambda_coeff))
        B[0][0] = 0
        #Compute the 2 matrix products for the w vector. Use the standardized matrix instead.
        productA = np.linalg.inv(np.matmul(X_matrixStandardized.transpose(), X_matrixStandardized) + ((11/2)*B))
        productB = np.matmul(X_matrixStandardized.transpose(), trainingSet_t)

        #Since this will be the last run, overwrite the matricies with their standardized counter-parts for plotting.
        trainingSet_x = trainingSetStandardized_x
        X_matrix = X_matrixStandardized

        validationSet_x = validationSetStandardized_x
        X_matrix_validation = X_matrixStandardized_validation
        

    else:

        #Compute the 2 matrix products for the w vector.
        productA = np.linalg.inv(np.matmul(X_matrix.transpose(), X_matrix))
        productB = np.matmul(X_matrix.transpose(), trainingSet_t)

    #Mulitply the 2 computed matrix products from above.
    w_vector = np.matmul(productA, productB)
    
    #Get the prediction values based on the validation set with our newly calculated w coefficients.
    y_test_prediction = np.dot(X_matrix_validation, w_vector)

    #Plot the functions vs validation x set.
    plt.plot(validationSet_x, y_test_prediction)
    plt.plot(validationSet_x, true_t)
    plt.plot(trainingSet_x, trainingSet_t, 'o')
    plt.plot(validationSet_x, validationSet_t, 'o')

    plt.legend(["Prediction Function", "True Function", "Training Set", "Validation Set"])
    plt.title("Function plots with M = " + str(i))

    plt.show()

    #Compute training and validation errors
    if i == 0:
        training_error[0] = 0
        validation_error[0] = 0
    else:
        y_train_prediction = np.dot(X_matrix, w_vector)
        diff = np.subtract(trainingSet_t, y_train_prediction)
        training_error[i] = (np.dot(diff.transpose(), diff)/i)

        diff = np.subtract(validationSet_t, y_test_prediction)
        validation_error[i] = (np.dot(diff.transpose(), diff)/i)

#Compute average squared error between validationSet_t (targets) and Ftrue.
diff = np.subtract(validationSet_t, true_t)
avgSqrdErr = np.dot(diff.transpose(), diff)/120

#Now plot training and validation errors vs M, as well as average squared error.
M_arr = np.arange(0,12,1)
plt.plot(M_arr, training_error, 'o')
plt.plot(M_arr, validation_error, 'o')
plt.axhline(y=avgSqrdErr)
plt.legend(["Training Error", "Validation Error", "Average Squared Error"])
plt.title("Training/Validation Errors vs M")
plt.show()

#Now find best lambda amongst 50000 in the range 0 - 0.5
lambda_list = np.arange(0,0.5,0.0001)
lowest_lambda = 0.5
highest_lambda = 0.5
validation_error_lowest = 100
training_error_highest = 0
lambda_list_size = lambda_list.shape[0]

validation_error_array = np.zeros(lambda_list_size)
training_error_array = np.zeros(lambda_list_size)

for i in range (lambda_list_size):

    #Create B Matrix based on set lambda
    B = np.diag(np.full(12, 2*lambda_list[i]))
    B[0][0] = 0

    #Compute the 2 matrix products for the w vector. Use the standardized matrix instead.
    productA = np.linalg.inv(np.matmul(X_matrixStandardized.transpose(), X_matrixStandardized) + ((11/2)*B))
    productB = np.matmul(X_matrixStandardized.transpose(), trainingSet_t)

    #Mulitply the 2 computed matrix products from above.
    w_vector = np.matmul(productA, productB)

    #Get the prediction values based on the validation set with our newly calculated w coefficients.
    y_test_prediction = np.dot(X_matrix_validation, w_vector)

    #Get Validation Error
    diff = np.subtract(validationSet_t, y_test_prediction)
    validation_error = (np.dot(diff.transpose(), diff)/120)
    validation_error_array[i] = validation_error

    #Get Training Error
    y_train_prediction = np.dot(X_matrix, w_vector)
    diff = np.subtract(trainingSet_t, y_train_prediction)
    training_error = (np.dot(diff.transpose(), diff)/12)
    training_error_array[i] = training_error

    if training_error > training_error_highest:
        training_error_highest = training_error
        highest_lambda = lambda_list[i]
    
    #Check if less then current lowest error. If so, update lowest validation error and lowest lambda variables
    if validation_error < validation_error_lowest:
        validation_error_lowest = validation_error
        lowest_lambda = lambda_list[i]

#Compute average squared error between validationSet_t (targets) and Ftrue.
diff = np.subtract(validationSet_t, true_t)
avgSqrdErr = np.dot(diff.transpose(), diff)/120

plt.plot(lambda_list, validation_error_array)
plt.plot(lambda_list, training_error_array)
plt.axhline(y=avgSqrdErr)
plt.legend(["Validation error", "Training Error", "Average Squared Error"])
plt.title("Training/Validation Errors vs Lambda for M=11")
plt.show()

print("Lambda which provides the lowest validation error: " + str(lowest_lambda))
print("Lambda which provides the highest training error: " + str(highest_lambda))
print(validation_error_lowest)


#Now lets plot Ftrue and Fm with our 2 lambda values
#Overfitting is occurring with values very close to zero, so we'll choose 0.0001
#It's very hard to tell where underfitting was occuring, but it seems like it was towards 0.5 boundary.
lambda_list = [0.0001, 0.5]

for i in range (len(lambda_list)):
    
     #Create B Matrix based on set lambda
    B = np.diag(np.full(12, 2*lambda_list[i]))
    B[0][0] = 0

    #Compute the 2 matrix products for the w vector. Use the standardized matrix instead.
    productA = np.linalg.inv(np.matmul(X_matrixStandardized.transpose(), X_matrixStandardized) + ((11/2)*B))
    productB = np.matmul(X_matrixStandardized.transpose(), trainingSet_t)

    #Mulitply the 2 computed matrix products from above.
    w_vector = np.matmul(productA, productB)
    print(w_vector)

    #Get the prediction values based on the validation set with our newly calculated w coefficients.
    y_test_prediction = np.dot(X_matrix_validation, w_vector)

    #Plot the functions vs validation x set.
    plt.plot(validationSet_x, y_test_prediction)
    plt.plot(validationSet_x, true_t)
    plt.plot(trainingSet_x, trainingSet_t, 'o')
    plt.plot(validationSet_x, validationSet_t, 'o')

    plt.legend(["Prediction Function", "True Function", "Training Set", "Validation Set"])
    plt.title("Function plots with M = 11 and Lambda = " + str(lambda_list[i]))

    plt.show()
