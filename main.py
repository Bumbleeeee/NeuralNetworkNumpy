import numpy as np
import data

batch_size = 60
num_epochs = 200

# TODO: make that graph thing that shows accuracy on the diagonal

x_train, y_train, x_test, y_test = data.get_data()
# training data is already flat

rng = np.random.default_rng()

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.where(x>0, x, 0)

def relu_prime(x):
    return np.where(x>0, 1, 0)

class Layer:
    def __init__(self, input_dim, num_neurons):
        self.weights = 0.1 * rng.random((input_dim, num_neurons)) - 0.05 #TODO: what should be the range here
        self.biases = 0.1 * rng.random(num_neurons) - 0.05

    def forward(self, prev_layer):
        return np.dot(prev_layer, self.weights) + self.biases



'''
class:
init should be hyperparams
then should have a function to create layers with input dim and number of neurons
then NN class has a forward function which does the forward pass for each layer (layers are a list perhaps)
question is how to keep a1, z1, etc around, or should they just be recalculated?
    could just return them (python passes references anyways) and pass them all into the backprop function
    could also store them as member variables of the layers or network
'''
class NeuralNetwork:
    def __init__(self, layer_dims: tuple, lr):
        self.learning_rate = lr
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(Layer(layer_dims[i], layer_dims[i+1]))


    def forward(self, x_data):
        z_list = []
        a_list = [x_data]
        for layer in self.layers:
            z = layer.forward(a_list[-1])
            a = sigmoid(z)
            z_list.append(z)
            a_list.append(a)

        return z_list, a_list

    def backward(self, z_list, a_list, y_data):
        deltas = [None] * len(self.layers)
        deltas[-1] = -1 * (y_data - a_list[-1]) * sigmoid_prime(z_list[-1])
        for i in range(len(self.layers)-2, -1, -1):
            deltas[i] = np.dot(deltas[i+1], self.layers[i+1].weights.T) * sigmoid_prime(z_list[i])

        for i in range(len(self.layers)):
            self.layers[i].weights -= np.dot(a_list[i].T, deltas[i]) * self.learning_rate
            self.layers[i].biases -= np.sum(deltas[i], axis=0) * self.learning_rate


nn = NeuralNetwork((784, 64, 16, 10), 0.01)

for epoch in range(num_epochs):
    # shuffle data
    indices = rng.permutation(60000)
    x_train = x_train[indices]
    y_train = y_train[indices]

    num_accurate = 0
    for j in range(0, 60000, batch_size):
        x_batch = x_train[j:j+batch_size]
        y_batch = y_train[j:j+batch_size]

        z_l, a_l = nn.forward(x_batch)

        # argmax returns index where max is found
        num_accurate += np.sum((np.argmax(a_l[-1], axis=1) == np.argmax(y_batch, axis=1)))

        nn.backward(z_l, a_l, y_batch)

    if (epoch+1) % 10 == 0:
        print(f"EPOCH {epoch+1}: accuracy = {num_accurate / 60000 * 100}%")


# ----------- TEST --------------
z_test, a_test = nn.forward(x_test)

num_accurate = np.sum((np.argmax(a_test[-1], axis=1) == np.argmax(y_test, axis=1)))

print(f"Test accuracy: {num_accurate / 10000 * 100}%")





# flatten image for first layer (input layer)
# need weights matrix to get between each layer (what are the values between?)
# then apply sigmoid to keep values between 0 and 1
# also have bias which we subtract before applying activation function
# each neuron in a layer has weights and biases to get from prev layer

# matrix of weights has all weights to get to a given neuron in a single row
#   mult by column vector of input neurons then add bias and apply sigmoid


# loss: sum of squares of differences between prediction and actual (y_train or y_test)
# take small step in negative gradient direction when training
# cost should be average of cost for all iters


'''
------------------- OLD CODE FOR POSTERITY AND UNDERSTANDING ------------------

z1 = layer1.forward(x_batch)
a1 = sigmoid(z1) # layer 1 activations
z2 = layer2.forward(a1)
a2 = sigmoid(z2)
z3 = output_layer.forward(a2)
a3 = sigmoid(z3)



delta_3 = -1 * (y_batch - a3) * sigmoid_prime(z3) # output is activations of last layer
delta_2 = np.dot(delta_3, output_layer.weights.T) * sigmoid_prime(z2)
delta_1 = np.dot(delta_2, layer2.weights.T) * sigmoid_prime(z1)

output_layer.weights -= np.dot(a2.T, delta_3) * learning_rate
layer2.weights -= np.dot(a1.T, delta_2) * learning_rate
layer1.weights -= np.dot(x_batch.T, delta_1) * learning_rate

# sum here because don't have the matrix mult to do the sum for us
output_layer.biases -= np.sum(delta_3, axis=0) * learning_rate
layer2.biases -= np.sum(delta_2, axis=0) * learning_rate
layer1.biases -= np.sum(delta_1, axis=0) * learning_rate

'''