import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# load MNIST data
data = pd.read_csv('train.csv')

# 5 rows x 785 columns
print(data.head())

# place pandas frame in numpy array and shuffle the data
data = np.array(data)
np.random.shuffle(data)

# 42000 rows (images) x 785 pixels (1 label, 784 pixel values)
m, n = data.shape
print(m,n)

# split into training and validation data - 80:20
# extract some of the rows & all of the columns
train_data = data[0:int(0.8*m), :]
val_data = data[int(0.8*m):m, :]

# Training dataset
# features: extract all of the rows and all but the first column & transpose. then scale down to between 0 and 1
X_train = train_data[:, 1:].T
X_train = X_train / 255.0
# labels: extract all of the rows and just the first column
Y_train = train_data[:, 0]

# Validation dataset
# features: extract all of the rows and all but the first column & transpose. then scale down to between 0 and 1
X_val = val_data[:, 1:].T
X_val = X_val / 255.0
# labels: extract all of the rows and just the first column
Y_val = val_data[:, 0]

# (784, 33600)
print(X_train.shape)
# (33600, )
print(Y_train.shape)
# (784, 8400)
print(X_val.shape)
# (8400, )
print(Y_val.shape)

# [layer 0] 784 neurons/nodes -> [layer 1] 10 neurons -> [layer 2] 10 neurons -> [layer 3] 1 neuron
# input layer -> hidden layer -> hidden layer -> output layer
# initialize weights in a [0,1] range. Then move range to [-0.5,0.5]. If all weights start positive, gradients flow asymmetrically
def initialize_parameters():
  W1 = np.random.rand(10, 784) - 0.5
  B1 = np.random.rand(10, 1) - 0.5
  W2 = np.random.rand(10, 10) - 0.5
  B2 = np.random.rand(10, 1) - 0.5
  return W1, B1, W2, B2

# Rectified linear unit: if <= 0, return 0; if > 0, return x
def ReLU(X):
  return np.maximum(X, 0)

# e^x/sigma(e^x-i)
def softmax_calculator(Z):
  return np.exp(Z) / sum(np.exp(Z))

# Convert label value (0-9) into one-hot encoded version
# Fancy indexing to avoid looping: Sets exactly one 1 per row to match the label. And transposes to (10 x m)
# Y is labels matrix (m, )
# Line 1: np.zeros creates an (m x 10) matrix of 0s
# Line 2: 
# np.arange creates array [0, 1, 2 ..., m-1] and specifies row
# Y is an array of length m and specifies column
# Set corresponding entries for (row, column) to 1
def one_hot_converter(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  return one_hot_Y.T

# A0 (784 x 1) = X (784 x 1)
# Z1 (10 x 1) = W1*A0 (10 x 784)*(784 x 1) + B1 (10 x 1)
# A1 (10 x 1) = RELU(Z1) (10 x 1)
# Z2 (10 x 1) = W2*A1(10 x 10)*(10 x 1) + B2 (10 x 1)
# A2 (10 x 1) = Y^ = SOFTMAX(Z2) (10 x 1)
def forward_propagation(W1, B1, W2, B2, X):
  Z1 = W1.dot(X) + B1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + B2
  A2 = softmax_calculator(Z2)
  return Z1, A1, Z2, A2

# Determine derivative of loss function wrt to weights/biases
# loss = categorical cross-entropy loss = - SIGMA(y_i * ln(p)) => -ln(p)

# binary cross-entropy loss: mean log loss = -1/n* SIGMA[y_i*ln(p) + (1-y_i) * ln(1-p)]
# Bernoulli distribution: models the result of a single bernoulli trial with 2 possible results - success/failure
# Probability mass function: mathematical function that gives the probability that a discrete random variable is exactly equal to a specific value. probabilities must be >0 and sum to 1
# In logistic regression, If y = 1, probability is p, if y = 0, probability is 1-p; PMF compact form for one observation = p^y * (1-p)^(1-y)
# For many observations, likelihood = product of individual probabilities. We want to maximize this, so that it's close to 1
# It's easier to maximize a sum, so we take a log. SIGMA (ln(p^y * (1-p)^(1-y))) = SIGMA(y*ln(p) + (1-y)ln(1-p))
# Convert to a minimization problem by taking negative; and divide by num observations to get average loss.

# Put simply, you're trying to get your predictions of a positive outcome (label value 1) to be as close to 1 as possible, across multiple observations. 
# Dealing with products is harder than addition, so take a sigma over the log of probabilities
# eg. prediction starts at 0.5; ln(0.5) = -0.69; you want it to get closer to 1; eg. ln(0.7) = -.35, ln(0.99) = -.01
# take the negative of that value and try to minimize it

# categorical cross-entropy-loss = -1/n* SIGMA[y * ln(p)]
# In BCE, you measure the distance between a single true label (0 or 1) and a predicted probability p => -[y_i*ln(p) + (1-y_i) * ln(1-p)]
# For CCE, the true label becomes a one-hot encoded vector eg. (0, 1, 0) for class 2 of 3. You then sum the loss across all classes: - SIGMA [y * ln(p)] which further simplifies to - ln(p)

# dL/dW2 = dL/dA2 * dA2/dZ2* dZ2/dW2
# = d(-ln(A2))/dA2 * d(e^Z2/SIGMA(e^Z2))/dZ2 * d(W2.dot(A1) + B2)/dW2
# = - Y/A2 * {A2(1-A2) for i=j; SIGMA(-A2_i*_A2_j) for i!=j} * A1
# = (A2 - Y) * A1^T
# = 1/m* (A2 - Y) * A1^T

# first term explained: d(ln(x))/dx = 1/x
# middle term explained: e^z_j/SIGMA(e^z_k)
# Because every Softmax output A2_j depends on every input logit Z2_i (due to the shared denominator), we must calculate how a change in one specific logit Z2_i affects all possible outputs A2_j
# quotient rule: differentiate ratio of 2 differentiable functions u/v = (u'v - uv')/v^2
# When i = j  => u' = e^z_i; v' = e^z_i => (e^z_i*S - e^z_i*e^z_i)/S^2 = A2_i(1-A2_i)
# When i != j => u' = 0; v' = e^z_i => (0*S - e^z_i*e^z_j)/S^2 = -A2_i*A2_j
# first term * middle term = -Y_i/A2_i* A2_i(1-A2_i) +  SIGMA(-Y_j/A2_j * -A2_i*A2_j)

# dL/dB2 = dL/dA2 * dA2/dZ2* dZ2/dB2
# = (A2 - Y) * 1
# = 1/m * SIGMA(A2 - Y)

# dL/dW1 = dL/dA1 * dA1/dZ1 * dZ1/dW1
# = dL/dA2 * dA2/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/dW1
# = dL/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/dW1
# = dL/dZ2 * d(W2.dot(A1) + B2)/dA1 * d(RELU(Z1))/dZ1 * d(W1.dot(X) + B1)/dW1
# = [dL/dZ2  * W2^T * 1 if Z1 > 0, otherwise 0]* X^T
# = [W2^T * dL/dZ2 {element-wise-product} Z1 > 0]* X^T
# = 1/m * W2^T (A2 - Y) {element-wise-product} Z1 > 0  * X^T

# dL/dB1 = dL/dA1 * dA1/dZ1* dZ1/dB1
# = 1/m * SIGMA(W2^T (A2 - Y) {element-wise-product} Z1 > 0)

def backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y):
  one_hot_Y = one_hot_converter(Y)
  
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  dB2 = 1 / m * np.sum(dZ2)
  
  dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
  dW1 = 1 / m * dZ1.dot(X.T)
  dB1 = 1 / m * np.sum(dZ1)
  return dW1, dB1, dW2, dB2

# update params 
def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
  W1 = W1 - learning_rate * dW1
  B1 = B1 - learning_rate * dB1
  W2 = W2 - learning_rate * dW2
  B2 = B2 - learning_rate * dB2
  return W1, B1, W2, B2

# the prediction is the index that has the highest value in a (10 x 1) array/vector
# axis = 0; scan across rows (down columns)
def get_predictions(A2):
  return np.argmax(A2, 0)

# num correct predictions / total predictions
def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size

# run training for 'iterations' number of epochs; each batch contains all the training data
def gradient_descent(X, Y, alpha, iterations):
  W1, B1, W2, B2 = initialize_parameters()

  for i in range(iterations):
    Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X)
    dW1, dB1, dW2, dB2 = backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y)
    W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)

    if (i%20)==0:
      print("Iteration number: ", i)
      print("Accuracy = ", get_accuracy(get_predictions(A2), Y))
  return W1, B1, W2, B2

# run training for 1000 epochs
W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 0.1, 1000)

# test with the 560th item in the validation set
val_index = 560

# None helps convert (784, ) to (784, 1)
Z1val, A1val, Z2val, A2val = forward_propagation(W1, B1, W2, B2, X_val[:, val_index, None])
print("Predicted label: ", get_predictions(A2val))
print("Actual label: ", Y_val[val_index])

image_array = X_val[:,val_index].reshape(28,28)
plt.imshow(image_array, cmap='gray')
plt.show()

# Validation accuracy
Z1val, A1val, Z2val, A2val = forward_propagation(W1, B1, W2, B2, X_val)
val_acc = get_accuracy(get_predictions(A2val), Y_val)
print("Validation accuracy = ", val_acc)