import math
from copy import deepcopy


class Node:
    def __init__(self, x):
        self.x = x
        self.output = None
        self.connections = []

    def calculate(self):
        input = 0
        for connection in self.connections:
            if connection.enabled:
                input += connection.weight * connection.origin.output
        self.output = self.activation_function(input)

    @staticmethod
    def activation_function(x):
        return 1 / (1+math.exp(-x))


class Connection:
    def __init__(self, origin, target):
        self.origin = origin
        self.target = target
        self.enabled = True
        self.weight = 0


class Network:
    def __init__(self, genome):
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []

        # these are node genes not nodes
        nodes = genome.nodes
        connections = genome.connections
        node_dict = {}

        # make the nodes
        for node_gene in nodes:
            node_dict[node_gene.innovation_numer] = deepcopy(node_gene)
            x = node_gene.x
            if x <= 0.1:
                self.input_nodes += node_gene
            elif x >= 0.9:
                self.output_nodes += node_gene
            else:
                self.hidden_nodes += node_gene
        # sort by depth
        self.hidden_nodes = sorted(self.hidden_nodes, key=lambda n: n.x)

        # establish the connections
        for connection_gene in connections:
            n1 = connection_gene.origin
            n2 = connection_gene.target

            node1 = node_dict[n1.innovation_number]
            node2 = node_dict[n2.innovation_number]

            connection = Connection(node1, node2)
            connection.weight = connection_gene.weight
            connection.enabled = connection_gene.enabled

            node2.connections += connection

    def calculate(self, input):
        # check proper inputs
        if len(input) != len(self.input_nodes):
            print("provided data doesn't fit")
            quit()

        # insert data into network
        for i in range(len(input)):
            self.input_nodes[i].output = input[i]
        # propagate
        for node in self.hidden_nodes + self.output_nodes:
            node.calculate()
        # return value of outputs
        return [node.output for node in self.output_nodes]