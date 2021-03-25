from AbstractMiniMax import MiniMax
import numpy as np


class TicTacToe(MiniMax):
    dimension = 3
    board = np.ones((dimension, dimension))
    max_ply = dimension*dimension

    def get_moves(self, game_state):
        return zip(np.nonzero(game_state))

    def score(self, game_state):
        for space in game_state, game_state.T:
            for row in space:
                if abs(row.sum()) == self.dimension: return self.current_player if row[0] == self.act

    def game_over(self, game_state):
        pass

    def update(self, game_state, move):
        game_state[move] = None

    def encode(self, game_state):
        return game_state.tostring()


