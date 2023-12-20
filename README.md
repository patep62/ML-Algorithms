# ML-Algorithms
I write several different Machine Learning techniques from scratch in Python, only utilizing Numpy.
These implementations served as educational resources to really understand the core concepts behind the way AI works.
This README will breifly outline each algorithm used and their results. A more detailed report is available for each algorithm in the doc folder.

## Linear Regression: Implementation of the basic linear regression model using gradient descent.
All data for testing and training for generated sythetically using a sin function. The data sets were pulled from the sin function with some added Gaussian noise to create variance.

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/ad74928e-700b-4d46-95e9-153230e8a54b)

The model was tested with varying capacities from 1 to 11, portraying the effects of underfitting and overfitting. Regularization is used at higher capacites in an attempt to reduce the overfitting.

![linRegM5](https://github.com/patep62/ML-Algorithms/assets/71285160/55c7dce6-37e3-4b5c-8649-a91e36f0545f)
![image](https://github.com/patep62/ML-Algorithms/assets/71285160/bbd9d16f-da20-4c6e-a9e1-c317c4089ffa)

We can start to see some overfitting here

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/45714d24-580c-4863-8176-3319115419c8)


High Capacity model, coupled with Regularization to reduce the effects of overfitting

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/2ecb8b2f-f201-4aad-a505-1ba244f71de2)


## Logistic Regression: Binary classification using logistic regression with gradient and stochastic descent.
Data for this implementation was pulled from the Wisconsin breast cancer data set, which is available in scikit-learn. The two classes are "malignant" (the positive class) and "benign" (the negative class), and 30 features are used per example.

Here I've pulled some visualizations of the learning rates for batch gradient descent and stochastic gradient descent.

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/0a24f5a5-6b2a-4975-a694-f8c6cd01d64f)

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/d475512b-4c46-4b4c-9bfb-18443c16443b)


## K-Nearest Neighbors (KNN): Non-parametric classification algorithm using 5-Fold Cross-Validation to determine the best k values.

The Boston Housing dataset was used to train the KNN model here, pulled online from the University of Toronto's website. The Hyperparameter, k, has significant impact on the training results, and in an effort to find the best value, Cross-Validation is used.
This is done by splitting the training set into folds of training/validation sets to train the hyperparameter on. Below is a visualization of the relationship between the Cross-Validation and training mean squared error (MSE).

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/b17e1f95-c847-45f6-b588-2d62551f8cdd)

## Neural Networks: Basic feedforward neural network with 2 hidden layers and customizable activation functions.

The Neural Network here was used to tackle another binary classification problem, backnote authentication. There are four features in this set, with each example representing a backnote. The Neural Network was then trained to predict authentic or forgery. 2 Hidden layers were used, with Sigmoid and ReLU used as the activation functions. Additionally, Cross-Entropy Loss was used to define error between predictions.

After writing the algorithm, the first goal is to find the optimal size ratios between the hidden layers. By limiting the max number of hidden units to 8, we can test various different combinations on a validation set to figure out the best one.
The results were compiled into the following table

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/302a6f2a-f371-4777-8011-c48266bee41f)

The relationship between the training/validation Learning Curves look like this for the chosen model, using stochastic gradient descent to save on computations.

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/3d1b0df6-8f3b-4bc4-8d59-f1f45c7f4c6b)


## K-Means Clustering: Unsupervised learning algorithm for clustering data points.
The final algorithm, and the only unsupervised model, was used to segment and compress images I found on google. Matplotlob's image reading library was used to convert an image into a 2D array of RGB pixel values. From here, euclidean distance was used to determine the clustering. Again, Cross-Validation is used here to portay the differences in k values. Although higher values will always yield better results, the runtime increases exponentially. The implemented algorithm is far from the most optimized, for example, vectorization through numpy was not fully leveraged and would have sped up the algorithm considerably.

The first Image used, Lake Moraine.

![moraine-lake](https://github.com/patep62/ML-Algorithms/assets/71285160/a800cbd6-4743-432b-9114-f0bb63ebc14f)

The following results were achieved using an early stopping mechanism to prevent unnecessary iterations past convergence.
![image](https://github.com/patep62/ML-Algorithms/assets/71285160/9fab2a11-99d9-43da-9c51-c57c81adaee6)

The following images were then generated with increasing k values.

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/a08a15da-3dd2-4272-b792-94897152964d)
![image](https://github.com/patep62/ML-Algorithms/assets/71285160/32c6b74e-5df0-4768-8cf2-c2f08311c8a9)

With anything past k = 20 being visually identical

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/126c9a7b-5d70-4128-8572-3a7358b19705)

And a second image, for fun.

![fox](https://github.com/patep62/ML-Algorithms/assets/71285160/4b3a2014-75e5-41ad-88ea-660e71572d1d)

![image](https://github.com/patep62/ML-Algorithms/assets/71285160/cc095ed2-1f50-4589-bb10-6f02fe9a91e4)
![image](https://github.com/patep62/ML-Algorithms/assets/71285160/e1a6c784-6c55-4a07-989d-d2a4dbe6175e)

And thats it! All the code is available within this repository, feel free to try them yourself! I had a ton of fun working on these and it was a great introduction into the world of machine learning.
