from copy import deepcopy

max_val = 100000
min_val = -max_val


class MiniMax:
    max_ply = 5
    current_player = None

    alpha_beta = True
    caching = True
    state_dict = True

    # abstract methods
    def get_moves(self, game_state):
        raise Exception("This is abstract")
        return 0

    def score(self, game_state):
        raise Exception("This is abstract")
        return 0

    def game_over(self, game_state):
        raise Exception("This is abstract")
        return 0

    def update(self, game_state, move):
        raise Exception("This is abstract")
        return 0

    # main algorithm
    def __call__(self, *args, **kwargs):
        self.activate(*args, **kwargs)

    def minimax(self, game_state, ply=0, a=min_val, b=max_val):
        if ply > self.max_ply or self.game_over(game_state): return self.score(game_state)
        maxing = ply % 2 == 0
        target_val = min_val if maxing else max_val
        best_play = min_val
        for play in self.get_moves(game_state):
            modified = self.update(deepcopy(game_state), play)
            if self.caching:
                cached_val = self.check_cache(game_state)
                value = self.minimax(modified, ply + 1, a, b) if cached_val is None else cached_val
            else:
                value = self.minimax(game_state, ply + 1, a, b)
            if (value > target_val) == maxing:
                target_val = value
                best_play = play
                if self.alpha_beta:
                    if (a < target_val) if maxing else (b < target_val):
                        if maxing: a = target_val
                        else: b = target_val
                    if b <= a: break
        return target_val if ply else best_play, target_val


    # transposition tables
    def check_cache(self, game_state):
        try: return self.state_dict[self.encode(game_state)]
        except: return None

    def encode(self, game_state):  # abstract
        hash(game_state)
