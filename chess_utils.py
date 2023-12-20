import chess
import numpy as np
import pandas as pd
import pickle
import torch
from typing import Callable


def pretty_print_state_stack(state: np.ndarray) -> None:
    """Given a state stack, print each state in a readable format."""
    piece_symbols = {1: "W", -1: "B", 0: "."}

    # Print the rows in reverse order
    for row in reversed(state):
        print(" ".join(piece_symbols[piece] for piece in row))


def board_to_random_state(board: chess.Board) -> np.ndarray:
    """Given a chess board object, return a 8x8 np.ndarray.
    Every square should be randomly assigned to 1, -1, or 0.
    This is to sanity check the linear probe.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""
    state = np.zeros((8, 8), dtype=int)
    for i in range(64):
        state[i // 8, i % 8] = np.random.choice([-1, 0, 1])

    return state


def board_to_piece_color_state(board: chess.Board) -> np.ndarray:
    """Given a chess board object, return a 8x8 np.ndarray.
    The 8x8 array should tell if each square is black, white, or blank.
    White is 1, black is -1, and blank is 0.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""
    state = np.zeros((8, 8), dtype=int)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            # Assign 1 for white pieces and -1 for black pieces
            state[i // 8, i % 8] = 1 if piece.color == chess.WHITE else -1

    return state


def board_to_piece_state(board: chess.Board) -> np.ndarray:
    """Given a chess board object, return an 8x8 np.ndarray.
    The 8x8 array should tell what piece is on each square. A white pawn could be 1, a black pawn could be -1, etc.
    Blank squares should be 0.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""

    # Mapping of chess pieces to integers
    piece_to_int = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }

    # Because state is initialized to all 0s, we only need to change the values of the pieces
    state = np.zeros((8, 8), dtype=int)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_value = piece_to_int[piece.piece_type]
            # Multiply by -1 if the piece is black
            if piece.color == chess.BLACK:
                piece_value *= -1
            state[i // 8, i % 8] = piece_value

    return state


def create_state_stack(
    moves_string: str, custom_board_to_state_fn: Callable[[chess.Board], np.ndarray]
) -> np.ndarray:
    """Given a string of PGN format moves, create an 8x8 np.ndarray for every character in the string."""

    board = chess.Board()
    initial_states = []
    count = 1

    # Scan 1: Creates states, with length = number of moves in the game
    initial_states.append(custom_board_to_state_fn(board))
    # Apply each move to the board
    for move in moves_string.split():
        try:
            count += 1
            # Skip move numbers
            if "." in move:
                board.push_san(move.split(".")[1])
            else:
                board.push_san(move)

            initial_states.append(custom_board_to_state_fn(board))
        except:
            # because all games are truncated to len 680, often the last move is partial and invalid
            # so we don't need to log this, as it will happen on most games
            break

    # if count % 100 == 0:
    #     pretty_print_state_stack(state)
    #     print("_" * 50)
    #     print(board)

    # Second Scan: Expand states to match the length of moves_string
    # For ;1.e4 e5 2.Nf3, ";1.e4" = idx 0, " e5" = idx 1, " 2.Nf3" = idx 2
    expanded_states = []
    move_index = 0
    for char in moves_string:
        if char == " ":
            move_index += 1
        expanded_states.append(initial_states[min(move_index, len(initial_states) - 1)])

    # expanded_states.append(initial_states[-1]) # The last element in expanded_states is the final position of the board.
    # Currently not using this as len(expanded_states) would be 1 greater than len(moves_string) and that would be confusing.
    return np.array(expanded_states)


def create_state_stacks(
    moves_strings: list[str],
    custom_board_to_state_fn: Callable[[chess.Board], np.ndarray],
) -> torch.Tensor:
    """Given a list of strings of PGN format moves, create a tensor of shape (len(moves_strings), 8, 8).
    custom_board_to_state is a function that takes a chess.Board object and returns a 8x8 np.ndarray for
    board state, or 1x1 for centipawn advantage."""
    state_stacks = []

    for board in moves_strings:
        state_stack = torch.tensor(
            create_state_stack(board, custom_board_to_state_fn)
        ).long()
        state_stacks.append(state_stack)

    # Convert the list of tensors to a single tensor
    final_state_stack = torch.stack(state_stacks)
    final_state_stack = final_state_stack.unsqueeze(0)  # Add a dimension for the modes
    # Currently, there is just one mode and it isn't necessary. For now, I'm maintaining the dimension for future use.
    return final_state_stack


def state_stack_to_one_hot(
    num_modes: int,
    num_rows: int,
    num_cols: int,
    min_val: int,
    max_val: int,
    device: torch.device,
    state_stack: np.ndarray,
) -> torch.Tensor:
    """Input shape: assert(state_stacks_all_chars.shape) == (modes, sample_size, game_length, rows, cols)
    Output shape: assert(state_stacks_one_hot.shape) == (modes, sample_size, game_length, rows, cols, one_hot_range)
    """
    range_size = max_val - min_val + 1

    # Initialize the one-hot tensor
    one_hot = torch.zeros(
        state_stack.shape[0],  # num modes
        state_stack.shape[1],  # num games
        state_stack.shape[2],  # num moves
        num_rows,
        num_cols,
        range_size,
        device=device,
        dtype=int,
    )

    for val in range(min_val, max_val + 1):
        one_hot[..., val - min_val] = state_stack == val

    return one_hot


def one_hot_to_state_stack(one_hot: torch.Tensor, min_val: int) -> np.ndarray:
    """Input shape: assert(probe_out.shape) == (modes, sample_size, num_white_moves, rows, cols, one_hot_range)
    Output shape: assert(state_stacks_probe_outputs.shape) == (modes, sample_size, num_white_moves, rows, cols)
    """
    indices = torch.argmax(one_hot, dim=-1)
    state_stack = indices.numpy() + min_val
    return state_stack


def find_dots_indices(moves_string: str) -> list[int]:
    """Returns a NumPy array of indices of every '.' in the string.
    This will hopefully provide a reasonable starting point for training a linear probe.
    """
    indices = [index for index, char in enumerate(moves_string) if char == "."]
    return indices


def encode_string(s: str) -> list[int]:
    """Encode a string into a list of integers."""
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    return [stoi[c] for c in s]


def decode_list(l: list[int]) -> str:
    """Decode a list of integers into a string."""
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    itos = meta["itos"]
    return "".join([itos[i] for i in l])
