import random
from Supervised.NeuralNetworks.BasicNN import NeuralNetwork, Neuron


class Population:
    def __init__(self, amount, layers, activations=(), mutation_rate=.1, survival_rate=.5):
        """
        A group of NN that will mimic evolution to train
        :param amount: the amount of NN in the population
        :param layers: the shape of each NN
        :param activations: optional param for the activations on each NN
        :param mutation_rate: how rapidly the NN change
        :param survival_rate: what percent survive each round
        """
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.population = []
        self.size = amount
        for i in range(amount):
            net = NeuralNetwork(layers, activations)
            self.population.append(net)

    def play_generation(self, scores, high_bad=False):
        # score_dict = {net: score for net, score in zip(self.population, scores)}
        self.sort_population(scores)
        self.population = self.population[::-1] if not high_bad else self.population
        self.population = self.population[:int(len(self.population)*self.survival_rate)]  # this is the culling
        self.repopulate()

    def sort_population(self, values):  # todo: change for quicksort
        sorted_list = []
        sorted_values = []
        for net, val in zip(self.population, values):
            insertion = self.binary_search(sorted_values, val)
            sorted_list.insert(insertion, net)
            sorted_values.insert(insertion, val)
        self.population = sorted

    def repopulate(self):
        new_nets = []
        while len(self.population) + len(new_nets) < self.size:
            new_nets.append(self.mutate_merge(random.choice(self.population), random.choice(self.population)))
        self.population.extend(new_nets)

    def mutate_merge(self, net1, net2):
        new_network = []
        for layer1, layer2 in zip(net1.network, net2.network):
            new_layer = []
            for neuron1, neuron2 in zip(layer1, layer2):
                new_weights = []
                for weight1, weight2 in zip(neuron1.weights, neuron2.weights):
                    new_weight = random.choice([weight1, weight2])
                    delta = new_weight * self.mutation_rate
                    new_weight += random.uniform(-delta, delta)
                    new_weights.append(new_weight)
                new_layer.append(Neuron(new_weights))
            new_network.append(new_layer)
        new_net = NeuralNetwork((1, 1))
        new_net.network = new_network
        new_net.activations = net1.activations
        return new_net

    @staticmethod
    def merge(net1, net2):
        new_network = []
        for layer1, layer2 in zip(net1.network, net2.network):
            new_layer = []
            for neuron1, neuron2 in zip(layer1, layer2):
                new_weights = []
                for weight1, weight2 in zip(neuron1.weights, neuron2.weights):
                    average_weight = (weight1 + weight2) / 2
                    new_weights.append(average_weight)
                new_layer.append(Neuron(new_weights))
            new_network.append(new_layer)
        new_net = NeuralNetwork((1, 1), net1.activations)
        new_net.network = new_network
        return new_net

    def mutate(self, net):
        for layer in net.network:
            for neuron in layer:
                new_weights = []
                for weight in neuron.weights:
                    delta = weight * self.mutation_rate
                    weight += random.uniform(-delta, delta)
                    new_weights.append(weight)
                neuron.weights = new_weights
        return net

    @staticmethod
    def binary_search(compare_list, val):
        if len(compare_list) == 0:
            return 0
        low_index = 0
        high_index = len(compare_list) - 1
        while low_index <= high_index:
            middle_index = round((high_index + low_index) / 2)
            if compare_list[middle_index] == val:
                return middle_index
            if val > compare_list[middle_index]:
                low_index = middle_index + 1
            elif val < compare_list[middle_index]:
                high_index = middle_index - 1
        return low_index
