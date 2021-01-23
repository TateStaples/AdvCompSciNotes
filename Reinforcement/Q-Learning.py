import random
import numpy as np
from general import GameState
from Supervised.NeuralNetworks.BasicNN import NeuralNetwork, flatten


class AbstractQLearner:
    def __init__(self, learn_rate=0.2, discount_rate=0.95, exploration_rate=0.1):
        """
        A base class that should never be created. Provides abstract and utility functions that get inherited
        :param learn_rate: the rate at which the program values new information over experience
        :param discount_rate: the amount that rewards fade as turns are taken (patience)
        :param exploration_rate: odds that learner will try things randomly (curiosity)
        """
        self.learning_rate = learn_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.state = None

    @staticmethod
    def weighted_average(data):  # for probabilistic scenarios
        """ Return the weighted average of list of tuples in format value, weight """
        sum_weight = 0
        sum_val = 0
        for val, weight in data:
            sum_weight += weight
            sum_val += weight * val
        return sum_val/sum_weight

    def random_action(self):
        """ test of whether to follow best action or explore something new. Based on Exploration Rate """
        return random.random() < self.exploration_rate

    def evaluate(self, state):  # abstract
        """
        Score how good a certain state is
        :param state: A gamestate object representing the game's current envirnment
        :return: a float value representing the quality of the state
        """
        raise Exception("This is an abstract function. You should not be here")

    def update_score(self, new_state, action, score):  # abstract
        """
        Tells the program to learn from the action that has been taken
        :param new_state: the resulting state of the action
        :param action: the index of the action that was taken
        :param score: the score of the action that was taken
        :return: None
        """
        raise Exception("This is an abstract function. You should not be here")

    def choose_action(self):
        """
        The single time step. Will either randomly explore or choose the best action to follow
        :return: None
        """
        if self.random_action():  # if exploring
            action, _ = random.choice(self.state.get_actions())
        else:
            action = None
            best_score = 0
            for a in self.state.get_actions():
                new_state, _ = self.state.update(a)
                s = self.evaluate(new_state)
                if s > best_score:
                    action = a
                    best_score = s
        new_state, reward = GameState.update(self.state, action)
        self.update_score(reward, new_state)
        self.state = new_state

    def _bellman(self, reward, new_state):
        """
        Evaluates what the new value of the state should be given current and previous data
        :param reward: how much the action taken benefited the agent this step
        :param new_state: the resulting state after the action was taken
        :return: A number that should be the new quality of the action
        """

        # Bellman Equation:
        # new quality = current quality + learning_rate * (reward + discount * next_state_quality)
        return (1 - self.learning_rate) * self.evaluate(self.state) + \
               self.learning_rate * (reward + self.discount_rate * self.evaluate(new_state))

    def play(self, start_state: GameState, time: int):
        """
        Main training loop
        :param start_state: The GameState where the agent starts the game
        :param time: how many actions the agent gets to train with
        :return: None
        """
        self.state = start_state
        for i in range(time):
            self.choose_action()


class StandardQLearner(AbstractQLearner):
    def __init__(self, action_shape: tuple, learn_rate=0.2, discount_rate=0.95, exploration_rate=0.1):
        """
        Uses a Q-board to learn about the proper actions to take in each scenario
        :param action_shape: a tuple representing the (amount_of_states, actions_per_state)
        :param learn_rate: the rate at which the program values new information over experience
        :param discount_rate: the amount that rewards fade as turns are taken (patience)
        :param exploration_rate: odds that learner will try things randomly (curiosity)
        """
        super(StandardQLearner, self).__init__(learn_rate, discount_rate, exploration_rate)
        self.Q = np.zeros(action_shape)

    def evaluate(self, state):
        return np.argmax(self.Q[state.index()])

    def update_score(self, new_state, action, score):
        self.Q[self.state] = self._bellman(score, new_state)


class DeepQLearner(AbstractQLearner):
    def __init__(self, model: NeuralNetwork, batch_size=100,
                 learn_rate=0.2, discount_rate=0.95, exploration_rate=0.1):
        """
        A Q-learning classes that uses a NN to evaluate board quality instead of storing everything
        :param model: the NN evaluation model to train
        :param batch_size: how many steps to wait before each training
        :param learn_rate: the rate at which the program values new information over experience
        :param discount_rate: the amount that rewards fade as turns are taken (patience)
        :param exploration_rate: odds that learner will try things randomly (curiosity)
        """
        super(DeepQLearner, self).__init__(learn_rate, discount_rate, exploration_rate)
        self.model = model
        self.batch_size = batch_size
        self._counter = 0
        self.stored_updates = []

    def evaluate(self, state):
        return self.model.predict(state.encode)

    def update_score(self, new_state, action, score):
        self._counter += 1
        update = (self.state, action, new_state, score)
        self.stored_updates.append(update)
        if self._counter % self.batch_size == 0:
            self._train()

    def _train(self):
        """
        Train the model on a batch of information that has been collected over the past several actions
        :return: None, but model will have updated
        """
        data, results = list(), list()
        for state, action, new_state, reward in self.stored_updates:
            data.append(state.encode)
            score = self._bellman(reward, new_state)
            blank_list = [0 for i in state.get_actions]
            blank_list[action] = score
            results.append(blank_list)
        self.model.train_network(data, results, l_rate=self.learning_rate, n_epoch=1)

