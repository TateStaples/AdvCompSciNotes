import random
from NEAT.Settings import *
from copy import deepcopy


def probability(prob):
    return prob < random.random()


def binary_insert(thing, seq: list):
    bottom = 0
    top = len(seq)
    while True:
        middle = round((bottom + top) / 2)
        item = seq[middle]
        if item == thing:
            seq.insert(middle, thing)
            return
        elif item > thing:
            bottom = middle
        else:
            top = middle - 1


class Gene:
    _innovation_number = 0
    
    def __init__(self):
        self.innovation_number = self._innovation_number
        self._innovation_number += 1

    def __hash__(self):
        return self.innovation_number


class NodeGene(Gene):
    def __init__(self, x=0, y=0):
        super(NodeGene, self).__init__()
        self.x = x
        self.y = y

    def __eq__(self, other):
        return type(other) is NodeGene and self.x == other.x and self.y == other.y

    def __repr__(self):
        pass


class ConnectionGene(Gene):
    def __init__(self, n1, n2):
        self.origin = n1
        self.target = n2
        self.enabled = True
        self.weight = random.random()
        self.replace_index = None
        super(ConnectionGene, self).__init__()

    def __repr__(self):
        return f"ConnectionGene(from={self.origin.innovation_number}, to={self.target.innovation_number}," \
               f"weight={self.weight}, enable={self.enabled}, innovation #{self.innovation_number})"

    def __eq__(self, other):
        return type(other) is ConnectionGene and other.origin == self.origin and other.target == self.target


class Genome:
    def __init__(self, neat):
        self.connections = set([])
        self.nodes = set([])
        self.neat = neat

    @staticmethod
    def sort(g1, g2):
        latest_innovation1 = g1.connections[-1] if len(g1.connections) > 0 else 0
        latest_innovation2 = g2.connections[-1] if len(g2.connections) > 0 else 0
        if latest_innovation2 > latest_innovation1:
            g1, g2 = g2, g1
        return g1, g2

    def distance(self, other):
        g1, g2 = self.sort(self, other)

        # create the accessing variables
        index_g1 = 0
        index_g2 = 0

        disjoint = 0  # things present in one and not another
        excess = 0  # how many more g1 has than g2
        weight_diff = 0  # average difference in weight of similar connections
        similar = 0  # how many nodes are in the same place

        while index_g1 < len(g1.connections) and index_g2 < len(g2.connections):
            gene1 = g1.connections[index_g1]  # these are connection genes
            gene2 = g2.connections[index_g2]

            in1 = gene1.innovation_number
            in2 = gene2.innovation_number

            if in1 == in2:  # if they are similar - same from and to
                similar += 1
                weight_diff += abs(gene1.weight - gene2.weight)
                index_g1 += 1
                index_g2 += 1
            elif in1 > in2:  # if in1 is earlier, there is a extra and move the 2nd on
                disjoint += 1
                index_g2 += 1
            else:  # same as previous but opposite
                disjoint += 1
                index_g1 += 1

        weight_diff /= max(1, similar)  # change from sum dif to average dif
        excess = len(g1.connections) - index_g1  # how many genes past g2
        n = max(len(g1.connections), len(g2.connections))  # lowers distance when things get bigger
        n = 1 if n < 20 else n

        return C1 * disjoint / n + C2 * excess / n + C3 * weight_diff / n

    def crossover(self, other):
        neat = self.neat
        g1, g2 = self.sort(self, other)
        new_genome = neat.empty_genome()
        i1 = 0
        i2 = 0

        while i1 < len(g1.connections) and i1 < len(g2.connections):
            gene1 = g1.connections[i1]
            gene2 = g2.connections[i2]

            in1 = gene1.innovation_number
            in2 = gene2.innovation_number

            if in1 == in2:  # if in both parents
                if probability(0.5):
                    new_genome.connections += deepcopy(gene1)
                else:
                    new_genome.connections += deepcopy(gene2)
            elif in1 > in2:  # if only in gene 2
                i2 += 1  # g2 is recessive so
            else:  # add
                i1 += 1
                new_genome.connections += deepcopy(gene1)

        # add all the extras
        while i1 < len(g1.connections):
            new_genome.connections += deepcopy(g1.connections[i1])
            i1 += 1

        # add all the nodes
        for connection_gene in new_genome.connections:
            new_genome.nodes += connection_gene.origin
            new_genome.nodes += connection_gene.target

        self.nodes = list(self.nodes)
        self.connections = list(self.connections)
        return new_genome

    def mutate(self):
        # have a chance to do each action
        if probability(PROBABILITY_MUTATE_LINK):
            self.mutate_link()
        if probability(PROBABILITY_MUTATE_NODE):
            self.mutate_nodes()
        if probability(PROBABILITY_MUTATE_WEIGHT_SHIFT):
            self.mutate_weight_shift()
        if probability(PROBABILITY_MUTATE_WEIGHT_RANDOM):
            self.mutate_weight_random()
        if probability(PROBABILITY_MUTATE_TOGGLE_LINK):
            self.mutate_link_toggle()

    def mutate_link(self):
        for i in range(100):  # try 100 times
            n1 = random.choice(self.nodes)
            n2 = random.choice(self.nodes)

            if n1 is None or n2 is None:
                continue

            if n1.x < n2.x:
                connection_gene = ConnectionGene(n1, n2)
            else:
                connection_gene = ConnectionGene(n2, n1)

            if connection_gene in self.connections:
                continue

            connection_gene = self.neat.get_connecting(connection_gene.origin, connection_gene.target)
            sign = random.random() * 2 - 1  # makes it neg or pos
            connection_gene.weight = sign * WEIGHT_RANDOM_STRENGTH  # amplify by predetermined amount

            binary_insert(connection_gene, self.connections)  # place in correct position
            return  # quit after a connection is added

    def mutate_nodes(self):
        connection_gene = random.choice(self.connections)
        if connection_gene is None:
            return
        n1, n2 = connection_gene.origin, connection_gene.target
        replace_index = self.neat.get_replace_index(n1, n2)
        if replace_index == 0:
            middle = self.neat.new_node()
            middle.x = (n1. x + n2.x) / 2
            middle.y = (n1.y + n2.y) / 2
            # self.neat.setReplaceIndex(n1, n2, middle.innovation_number)
        else:
            middle = self.neat.node_at(replace_index)

        con1 = self.neat.get_connecting(n1, middle)
        con2 = self.neat.get_connecting(middle, n2)

        con1.weight = 1
        con2.weight = connection_gene.weight
        con2.enabled = connection_gene.enabled

        self.connections.remove(connection_gene)
        self.connections += con1
        self.connections += con2

        self.nodes += middle

    def mutate_weight_shift(self):
        connection_gene = random.choice(self.connections)
        if connection_gene is not None:
            sign = random.random() * 2 - 1
            connection_gene.weight = connection_gene.weight + sign * WEIGHT_SHIFT_STRENGTH

    def mutate_weight_random(self):
        connection_gene = random.choice(self.connections)
        if connection_gene is not None:
            sign = random.random() * 2 - 1
            connection_gene.weight = sign * WEIGHT_RANDOM_STRENGTH

    def mutate_link_toggle(self):  # toggle random connection
        connection_gene = random.choice(self.connections)
        if connection_gene is not None:
            connection_gene.enabled = not connection_gene.enabled