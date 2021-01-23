from .Creatures import *
from .Genes import *
from copy import deepcopy
from .Settings import SURVIVORS


# todo: make this simpler

class NEAT:
    def __init__(self, input_size, output_size, creatures):
        # initialize data
        self.input_size = None
        self.output_size = None
        self.max_creatures = None
        self.connections = []
        self.nodes = []
        self.creatures = []
        self.species = []
        # start of the program
        self.reset(input_size, output_size, creatures)

    def empty_genome(self):
        # generate a blank genome
        g = Genome(self)

        # add the input and output neurons
        for i in range(self.input_size + self.output_size):
            g.nodes.add(self.node_at(i+1))
        return g

    def reset(self, input_size, output_size, creatures):
        # establish topology parameters
        self.input_size = input_size
        self.output_size = output_size
        self.max_creatures = creatures

        # reset previous data
        self.connections.clear()
        self.nodes.clear()
        self.creatures.clear()
        # self.species.clear()

        # create input nodes
        for i in range(input_size):
            n = self.new_node()
            n.x = 0.1
            n.y = (i+1) / input_size + 1

        # create output nodes
        for i in range(output_size):
            n = self.new_node()
            n.x = 0.9
            n.y = (i + 1) / input_size + 1

        # create new networks
        for i in range(creatures):
            c = Creature()
            c.genome = self.empty_genome()
            self.creatures += c

    def get_connecting(self, node1, node2):
        # check each connection for pre-existing
        for connection in self.connections:
            if connection.origin is node1 and connection.target is node2:
                # if found, return a copy
                return deepcopy(connection)
        # else make one and return a copy
        c = ConnectionGene(node1, node2)
        self.connections += c
        return deepcopy(c)

    def get_replace_index(self, n1, n2):
        for connection in self.connections:
            if connection.origin == n1 and connection.target == n2:
                return connection.replace_index
        return 0

    def new_node(self):
        n = NodeGene()
        self.nodes.append(n)
        return n

    def node_at(self, i):
        if i <= len(self.nodes):  # if it is in existance
            return self.nodes[i-1]  # return it, using higher index system
        return self.new_node()

    def evolve(self):
        # do one cycle of evolution
        self.create_species()
        self.cull()
        self.remove_extinct_species()
        self.reproduce()
        self.mutate()
        for c in self.creatures:
            c.generate_calculator()

    def print_species(self):
        print("#------------------------------#")
        for species in self.species:
            print(f"{species} {species.score} {species.size()}")

    def reproduce(self):
        for c in self.creatures:
            species = random.choice[self.species]  # todo: replace with weighted average
            c.genome = species.breed()  # make a member of species
            species.force_put(c)  # add creature into species

    def mutate(self):
        # mutate each creature
        for c in self.creatures:
            c.mutate()

    def remove_extinct_species(self):
        for species in reversed(self.species):  # for each species
            if not species.size():  # if there are no member
                species.go_extinct()  # go extinct
                self.species.remove(species)  # remove from list

    def create_species(self):
        # clear out species
        for species in self.species:
            species.reset()

        for c in self.creatures:  # for each creature
            if c.species is None:  # if it does not belong to a species
                for species in self.species:
                    if species.put(c):  # check if it is a member
                        break
                else:  # else create a new species for it
                    self.species += Species(c)

    def cull(self):
        # for each species
        for species in self.species:
            # leave on a certain percentage alive
            species.kill(1-SURVIVORS)


if __name__ == '__main__':
    # create network
    n = NEAT(10, 1, 1000)

    # create random data
    inputs = [random.random() for i in range(10)]

    # for 100 turns
    for i in range(100):
        # for each creature
        for c in n.creatures:
            # train on data
            score = c.calculate(inputs)[0]
            c.score = score
        # and do stuff
        n.evolve()
        # n.printSpecies()

    # print the species
    for c in n.creatures:
        for gene in c.genome.connections.data:
            print(gene.innovation_number, end=" ")