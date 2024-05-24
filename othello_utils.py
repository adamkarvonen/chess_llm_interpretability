import torch as t
from datasets import load_dataset
from othello_engine_utils import OthelloBoardState, stoi, itos


def board_state_to_RRC(board_state, flip: int = 1):
    board_state = t.tensor(board_state, dtype=t.int8)
    board_state *= flip
    one_hot = t.zeros((8, 8, 3), dtype=t.int8)
    one_hot[..., 0] = (board_state == -1).int()
    one_hot[..., 1] = (board_state == 0).int()
    one_hot[..., 2] = (board_state == 1).int()
    return one_hot


# TODO Remove duplicated logic from these functions
def games_batch_to_state_stack_BLRRC(batch_str_moves):
    """Sequences of moves (dataset format) to state stack (one-hot) of shape (seq_len, 8, 8, 3)"""
    game_stack = []
    for game in batch_str_moves:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for move in game:
            board.umpire(move)
            one_hot = board_state_to_RRC(board.state)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_valid_moves_BLRRC(batch_str_moves):
    """Sequences of moves (dataset format) to state stack of valid moves"""
    game_stack = []
    for game in batch_str_moves:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            moves_board = t.zeros(8, 8, 1, dtype=t.int8)
            board.umpire(move)
            valid_moves_list = board.get_valid_moves()
            for move in valid_moves_list:
                moves_board[move // 8, move % 8] = 1
            states.append(moves_board)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_state_stack_mine_yours_BLRRC(batch_str_moves):
    """Sequences of moves (dataset format) to state stack (one-hot) of shape (seq_len, 8, 8, 3)"""
    game_stack = []
    for game in batch_str_moves:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 1:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_RRC(board.state, flip)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


othello_functions = [
    games_batch_to_state_stack_BLRRC.__name__,
    games_batch_to_state_stack_mine_yours_BLRRC.__name__,
    games_batch_to_valid_moves_BLRRC.__name__,
]


def get_othello_even_list_indices(tokens_list: list[int]) -> list[int]:
    """"""
    max_len = len(tokens_list)
    return [i for i in range(max_len) if i % 2 == 0]


def get_othello_all_list_indices(tokens_list: list[int]) -> list[int]:
    """"""
    max_len = len(tokens_list)
    return [i for i in range(max_len)]
