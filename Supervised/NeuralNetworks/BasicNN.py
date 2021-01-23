from random import *
from math import *
from copy import deepcopy


class Neuron:
    def __init__(self, weights):
        self.weights = weights
        self.output = 0
        self.delta = 0

    def __repr__(self):
        return f"weights: {self.weights}, output:{self.output}, delta: {self.delta}"


class NeuralNetwork:
    # create the layers
    def __init__(self, layers, activations=None):
        if activations is None or len(activations) != len(layers)-1:  # this lets the user choose activation for layers
            self.activations = [self.sigmoid for i in range(len(layers)-1)]
        else:
            self.activations = activations
        self.network = []  # the actual network as a list of layers
        self.amount_of_inputs = layers[0]  # shows how many input there should be into this network
        self.amount_of_outputs = layers[-1]  # amount of outputs this network has
        previous_layer = None
        for amount_of_nodes in layers:
            if previous_layer is not None:
                # connects each neuron to each of the previous ones
                layer = [Neuron([uniform(-1, 1) for i in range(previous_layer + 1)]) for i in range(amount_of_nodes)]
                self.network.append(layer)
            previous_layer = amount_of_nodes

    # Calculate neuron activation for an input
    @staticmethod
    def activate(weights, inputs):
        # print(weights)
        activation = weights[-1]  # this last one is the bias
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]  # sums all the incoming values
        return activation

    # Forward propagate input to a network output
    def forward_propagate(self, inputs):
        for l, layer in enumerate(self.network):
            new_inputs = []  # this is the outputs of the current layer
            for neuron in layer:
                activation = self.activate(neuron.weights, inputs)  # the strength of neuron
                neuron.output = self.activations[l](activation)  # process strength with activation function
                new_inputs.append(neuron.output)
            inputs = new_inputs  # move to the next layer
        return inputs

    # updates the network
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):  # starts from output and moves back (look at name)
            layer = self.network[i]  # layer working with
            errors = list()  # list of how far off each neuron is
            if i != len(self.network) - 1:  # if not the output layer
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:  # for neuron this one leads to
                        error += (neuron.weights[j] * neuron.delta)  # error is the strength time delta
                    errors.append(error)
            else:  # output layer
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron.output)  # the difference between predicted and actual
            for j in range(len(layer)):
                neuron = layer[j]
                neuron.delta = errors[j] * self.activations[i](neuron.output, derivative=True)

    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i in range(len(self.network)):  # iterates through layers
            inputs = row  # all weights (excludes the bias)
            if i != 0:  # not input layer
                inputs = [neuron.output for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron.weights[j] += l_rate * neuron.delta * inputs[j]  # update each weight
                neuron.weights[-1] += l_rate * neuron.delta  # update bias

    # trains teh network with large datasets
    def train_network(self, train_data, train_results, l_rate, n_epoch, refine_to=None):
        refining = refine_to is not None
        if refining:
            amount_of_trials = n_epoch * len(train_data)
            desired_shift = l_rate - refine_to
            # delta = l_rate**(1.0/n_epoch)
            delta = desired_shift / (n_epoch)
        for epoch in range(n_epoch):  # amount of times to train on set
            sum_error = 0
            for r, row in enumerate(train_data):  # for each datum
                if r % (len(train_data)//10) == 0:
                    print(f"%{len(train_data)//10}0 percent through")
                outputs = self.forward_propagate(row)  # make a prediction
                expected = train_results[r]  # this is what you should have gotten
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)  # update the network
                self.update_weights(row, l_rate)  # read adjust the weights
            print(f'epoch={epoch}, learning rate={l_rate}, total error={sum_error}')
            if refining:
                # l_rate *= delta
                l_rate -= delta

    # get the strength for each - just a different word for forward propogate
    def get_chances(self, inputs):
        return self.forward_propagate(inputs)

    # Make a prediction with a network - get the highest guess
    def predict(self, input):
        outputs = self.forward_propagate(input)
        return outputs.index(max(outputs))

    # get percent accuracy of network
    def evaluate(self, test_data, test_labels, loss_function=None):
        total_data = len(test_data)
        guessed_correctly = 0
        for data, label in zip(test_data, test_labels):
            guess = self.forward_propagate(data)
            sum_error = 0
            for guessed, actual in zip(guess, label):
                sum_error += (guessed - actual) ** 2
            guessed_correctly += 1 if guess.index(max(guess)) == label.index(max(label)) else 0
        percentage = (guessed_correctly/total_data) * 100
        print(f"The network had an accuracy of {guessed_correctly}/{total_data} or {percentage}")
        return percentage

    @staticmethod
    def liner_unit(x, derivative=False):  # most basic, quick to lear, but cant do complicated
        return x if not derivative else 1

    @staticmethod
    def sigmoid(x, derivative=False):  # sigmoid (an activation function)
        try:
            return 1 / (1 + e**(-x)) if not derivative else x * (1.0 - x)
        except OverflowError:
            return 0 if derivative else 0 if x < 0 else 1

    @staticmethod
    def tanh(x, derivative=False):  # performs better than sigmoid (-1 to 1)
        if derivative:
            return (cosh(x)**2 - sinh(x)**2) / cosh(x)**2
        return tanh(x)

    @staticmethod
    def relu(x, derivative=False):  # rectified linear unit, good for deep / complicated networks
        if derivative:
            return 0 if x < 0 else 1
        return x if x > 0 else 0

    @staticmethod
    def leaky_relu(x, derivative=False, neg_slope=0.1):  # variant that prevents dead neurons
        if derivative:
            return neg_slope if x < 0 else 1
        return x*neg_slope if x < 0 else x

    @staticmethod
    def hardmax(x, deriv=False):
        if deriv:
            return 0
        return 0 if x < 0 else 1

    # used to represent the network
    def print_layers(self):
        for l, layer in enumerate(self.network, start=1):
            for n, neuron in enumerate(layer, start=1):
                print(f"neuron {n} of layer {l} = {neuron}")
            print()

    # used to save network to a file
    def write_to_file(self, file):
        for layer in self.network:
            for neuron in layer:
                for weight in neuron.weights:
                    file.write(f"{weight}")
                    if weight != neuron.weights[-1]:
                        file.write(",")
                file.write("\n")
            file.write("\n")

    # used to apply a saved network
    def insert_weights_from_file(self, file):
        layer = 0
        layer_weights = []
        for line in file:
            line = line.strip()
            if line != "":
                weights = line.split(",")
                print(weights)
                weights = [float(i.strip(",")) for i in weights]
                layer_weights.append(weights)
            else:
                for neuron, weight in zip(self.network[layer], layer_weights):
                    neuron.weights = weight
                layer += 1
                layer_weights = []

    # attaches a neural network to the output of the first
    def extend(self, other_net):
        network1 = deepcopy(self.network)
        network2 = deepcopy(other_net.network)
        new_net = network1 + network2
        activations1 = self.activations
        activations2 = other_net.activations
        new_activations = activations1 + activations2
        new_network = NeuralNetwork(())
        new_network.network = new_net
        new_network.activations = new_activations
        return new_network


def flatten(the_list):
    new_list = []
    for thing in the_list:
        if type(thing) == list:
            thing = flatten(thing)
            new_list.extend(thing)
        else:
            new_list.append(thing)
    return new_list
