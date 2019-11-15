import numpy as np
import keras
import keras.datasets
from keras.datasets import cifar10
from CleanNN import layer_sizes, initialize_parameters, backward_propagation, forward_propagation, compute_cost, update_parameters, nn_model, predict

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

### The shape of input samples and their labels ###
shape_X = x_train.shape
shape_Y = y_train.shape
m = y_train.shape[0]  # training set size

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('Size of training samples is = %d' % (m))




parameters = initialize_parameters(3, 5, 1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = nn_model(x_train.T, y_train.T, n_h = 5, num_iterations = 10000, print_cost=True)

plot_decision_boundary(lambda x: predict(parameters, x_train.T), x_train, y_train)
plt.title("Decision Boundary for hidden layer size " + str(5))

# Print accuracy
predictions = predict(parameters, x_train)
print ('Accuracy: %d' % float((np.dot(y_train,predictions.T) + np.dot(1-y_train,1-predictions.T))/float(y_train.size)*100) + '%')

# Testing different configuration of the network for the given data.

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50, 100]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(x_train, y_train, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x_train.T), x_train, y_train)
    predictions = predict(parameters, x_train)
    accuracy = float((np.dot(y_train,predictions.T) + np.dot(1-y_train,1-predictions.T))/float(y_train.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


# SINGLE hidden layer, 10 way softmax output, (for 10 classes)
# tanh activation for hidden layers and sigmoid for output_type
# COMPUTE CROSS ENTROPY loss
# forward and backward_propagation
