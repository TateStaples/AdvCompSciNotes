import math

# magnitude of mutations
WEIGHT_SHIFT_STRENGTH = 0.3
WEIGHT_RANDOM_STRENGTH = 1

# mutation changes
PROBABILITY_MUTATE_LINK = 0.01
PROBABILITY_MUTATE_NODE = 0.1
PROBABILITY_MUTATE_WEIGHT_SHIFT = 0.02
PROBABILITY_MUTATE_WEIGHT_RANDOM = 0.02
PROBABILITY_MUTATE_TOGGLE_LINK = 0

# i dont know where this is used
MAX_NODES = math.pow(2, 20)

# distance calculation settings
C1 = C2 = C3 = 1

# how far seperates species
CP = 4

# what percent of species survives culling
SURVIVORS = 0.8