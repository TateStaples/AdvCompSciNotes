import pygame
import numpy as np
import time
pygame.init()

windowWidth = 1200
windowHeight = 700
window = pygame.display.set_mode((windowWidth, windowHeight))


max_val = 1000000
min_val = -1000000
best_score = 1000

RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

board = np.zeros((6, 7))


def list_in(sub, super):
    for start_index in range(len(super) - len(sub) + 1):
        _slice = super[start_index : start_index + len(sub)]
        if list(_slice) == sub:
            return True
    return False


def draw_board():
    window.fill(WHITE)
    pygame.draw.rect(window, BLUE, [100, 100, 1000, 550])  # board
    for row_count, row in enumerate(board):
        for col, spot in enumerate(row):
            color = {0: WHITE, 1: RED, -1: YELLOW}[spot]
            pygame.draw.circle(window, color, (225 + 125 * col, 150 + row_count * 80), 30) # spots


def move(a_board, col, player):
    column = a_board[:, col]
    if column.all(): return a_board
    column[column.nonzero()[0][0]-1 if column.any() else len(column)-1] = player
    return a_board


def board_state(the_board, player):
    not_player = player * -1

    # check horizontal wins
    for space in the_board, the_board.T:
        for piece in space:
            if list_in([player] * 4, piece): return 1
            if list_in([not_player] * 4, piece): return -1

    flipped = np.fliplr(the_board)
    for diag in range(-2, 4):
        if list_in([player] * 4, the_board.diagonal(diag)): return 1
        if list_in([player] * 4, flipped.diagonal(diag)): return 1
        if list_in([not_player] * 4, the_board.diagonal(diag)): return -1
        if list_in([not_player] * 4, flipped.diagonal(diag)): return -1

    return 0


def game_over(the_board):
    state = board_state(the_board, 1)
    if state:
        return state
    return the_board[0].all()


board_values = np.array([
        [3, 4, 5, 7, 5, 4, 3],
        [4, 6, 8, 10, 8, 6, 4],
        [5, 8, 11, 13, 11, 8, 5],
        [5, 8, 11, 13, 11, 8, 5],
        [4, 6, 8, 10, 8, 6, 4],
        [3, 4, 5, 7, 5, 4, 3]
    ])


def evaluate_board(the_board, player):
    state = board_state(the_board, player)
    if state != 0: return state * best_score

    score = (the_board * board_values * player).sum()
    return score


def get_moves(the_board):
    return np.where(the_board[0] == 0)[0]


def encode(the_board):
    return the_board.tostring()


scored_states = dict()
final_states = dict()
def check_cache(the_board):
    encoded = encode(the_board)
    if encoded in scored_states:
        return scored_states[encoded], False
    if encoded in final_states:
        return final_states[encoded], True
    return None, None


def Mini_Max(the_board, player, current_ply, max_ply, og, alpha=min_val, beta=max_val):  # alpha is branch max and beta is branch min
    caching = False
    alpha_beta = True
    if current_ply >= max_ply or game_over(the_board): return evaluate_board(the_board, og)
    other = player * -1
    maxing = current_ply % 2 == 0
    target_val = min_val if maxing else max_val
    best_play = -1
    for play in get_moves(the_board):
        modified = move(the_board.copy(), play, player)
        value = Mini_Max(modified, other, current_ply + 1, max_ply, og, alpha, beta)
        if value == (best_score if maxing else -best_score): return value
        if caching:
            cached_val, _final = check_cache(modified)
            if cached_val is None:
                value, _final = Mini_Max(modified, other, current_ply + 1, max_ply, og, alpha, beta)
            else:
                value = cached_val
            scored_states[encode(modified)] = value
        if (value > target_val) == maxing:
            target_val = value
            best_play = play
            if alpha_beta and not caching:
                if (alpha < target_val) if maxing else (beta < target_val):
                    if maxing: alpha = target_val
                    else: beta = target_val
                if beta <= alpha:
                    break
    if current_ply == 0:
        global board
        board = move(board, best_play, og)
    return target_val


def ai_move(player):
    Mini_Max(board, player=player, current_ply=0, max_ply=5, og=player, alpha=min_val, beta=max_val)


def human_move(player):
    if player == 1:
        color = RED
    else:
        color = YELLOW
    global board
    clicked = None
    while clicked is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked = (pygame.mouse.get_pos()[0] - 100) // 140
                if clicked < 0 or clicked > 6:
                    clicked = None

        mouse_location = pygame.mouse.get_pos()[0]
        draw_board()
        pygame.draw.circle(window, color, (mouse_location, 50), 30)
        pygame.display.update()
    board = move(board, clicked, player)


def main():
    global board, nodes, alpha_cut, cache_cut
    player = 1
    primary_player = "human"
    secondary_player = "ai"
    draw_board()
    while not game_over(board):
        _type = primary_player if player == 1 else secondary_player
        if _type == "human":
            human_move(player)
        else:
            ai_move(player)

        status = board_state(board, 1)
        draw_board()
        pygame.display.update()
        if status:
            break
        scored_states.clear()
        player *= -1
    answers = ["draw", "Player 1 wins", "Player 2 wins"]
    print(answers[status])


if __name__ == '__main__':
    main()

    # this is so the app doesn't quit when over
    pygame.display.update()
    while True:
        pygame.time.delay(1000)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
