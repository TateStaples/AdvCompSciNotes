from NEAT.Network import Network
from NEAT.Settings import CP
import random


class Creature:
    def __init__(self):
        self.score = None
        self.species = None
        self.genome = None
        self.calculator = None

    def get_calculator(self):
        self.calculator = Network(self.genome)

    def calculate(self, input):
        if self.calculator is None:
            self.get_calculator()
        return self.calculator.calculate(input)

    def distance(self, other):
        return self.genome.distance(other.genome)

    def mutate(self):
        self.genome.mutate()


class Species:
    def __init__(self, representative):
        self.creatures = []
        self.representative = representative
        self.score = 0
        representative.species = self

    def put(self, creature):
        if creature.distance(self.representative) < CP:
            creature.species = self
            self.creatures += self
            return True
        return False

    def force_put(self, creature):
        creature.species = self
        self.creatures += creature

    def go_extinct(self):
        for creature in self.creatures:
            creature.species = None
        self.creatures.clear()

    def evaluate_score(self):
        self.score = sum([creature.score for creature in self.creatures]) / len(self.creatures)

    def reset(self):
        representative = random.choice(self.creatures)
        self.go_extinct()
        self.creatures += representative
        self.representative = representative
        representative.species = self
        self.score = 0

    def kill(self, percentage):
        self.creatures = sorted(self.creatures, key=lambda x: x.score)
        amount = int(percentage*self.size())
        for i in range(amount):
            creature = self.creatures.pop(0)
            creature.species = None

    def breed(self):
        c1, c2 = random.choice(self.creatures), random.choice(self.creatures)
        if c1.score > c2.score:
            return c1.genome.crossover(c2.genome)
        return c2.genome.crossover(c1.genome)

    def size(self):
        return len(self.creatures)