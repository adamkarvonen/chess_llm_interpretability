import chess
import numpy as np
import pandas as pd
import pickle
import torch
from torch.nn import functional as F
from typing import Callable, Optional


# Mapping of chess pieces to integers
PIECE_TO_INT = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

INT_TO_PIECE = {value: key for key, value in PIECE_TO_INT.items()}
PIECE_TO_ONE_HOT_MAPPING = {
    -6: 0,
    -5: 1,
    -4: 2,
    -3: 3,
    -2: 4,
    -1: 5,
    0: 6,
    1: 7,
    2: 8,
    3: 9,
    4: 10,
    5: 11,
    6: 12,
}
ONE_HOT_TO_PIECE_MAPPING = {
    value: key for key, value in PIECE_TO_ONE_HOT_MAPPING.items()
}


def pretty_print_state_stack(state: np.ndarray) -> None:
    """Given a state stack, print each state in a readable format.
    The problem is that chess boards row 0 begin at the bottom, while in state stack (and any array in general),
    row 0 begins at the top. This is why we reverse the state stack before printing it.
    """
    piece_symbols = {1: "W", -1: "B", 0: "."}

    # Print the rows in reverse order
    for row in reversed(state):
        print(" ".join(piece_symbols[piece] for piece in row))


def board_to_random_state(
    board: chess.Board, skill: Optional[int] = None
) -> np.ndarray:
    """Given a chess board object, return a 8x8 np.ndarray.
    Every square should be randomly assigned to 1, -1, or 0.
    This is to sanity check the linear probe.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""
    state = np.zeros((8, 8), dtype=int)
    for i in range(64):
        state[i // 8, i % 8] = np.random.choice([-1, 0, 1])

    return state


def board_to_skill_state(board: chess.Board, skill: float) -> np.ndarray:
    """Given a chess board object, return a 1x1 np.ndarray.
    The 1x1 array should tell what skill level the player is."""
    state = np.zeros((1, 1), dtype=float)
    state[0][0] = skill

    return state


def board_to_piece_color_state(
    board: chess.Board, skill: Optional[int] = None
) -> np.ndarray:
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


def board_to_piece_state(board: chess.Board, skill: Optional[int] = None) -> np.ndarray:
    """Given a chess board object, return an 8x8 np.ndarray.
    The 8x8 array should tell what piece is on each square. A white pawn could be 1, a black pawn could be -1, etc.
    Blank squares should be 0.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""

    # Because state is initialized to all 0s, we only need to change the values of the pieces
    state = np.zeros((8, 8), dtype=int)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_value = PIECE_TO_INT[piece.piece_type]
            # Multiply by -1 if the piece is black
            if piece.color == chess.BLACK:
                piece_value *= -1
            state[i // 8, i % 8] = piece_value

    return state


def state_stack_to_chess_board(state: torch.Tensor) -> chess.Board:
    """Given a state stack, return a chess.Board object.
    WARNING: The board will not include any information about whose turn it is, castling rights, en passant, etc.
    For this reason, pgn_string_to_board is preferred."""
    board = chess.Board(fen=None)
    for row_idx, row in enumerate(state):
        for col_idx, piece in enumerate(row):
            if piece != 0:
                piece_type = abs(piece)
                color = chess.WHITE if piece > 0 else chess.BLACK
                board.set_piece_at(
                    chess.square(col_idx, row_idx), chess.Piece(piece_type, color)
                )
    return board


def pgn_string_to_board(pgn_string: str) -> chess.Board:
    """Convert a PGN string to a chess.Board object.
    We are making an assumption that the PGN string is in this format:
    ;1.e4 e5 2. or ;1.e4 e5 2.Nf3"""
    board = chess.Board()
    for move in pgn_string.split():
        if "." in move:
            move = move.split(".")[1]
        if move == "":
            continue
        board.push_san(move)
    return board


def create_state_stack(
    moves_string: str,
    custom_board_to_state_fn: Callable[[chess.Board], np.ndarray],
    skill: Optional[float] = None,
) -> np.ndarray:
    """Given a string of PGN format moves, create an 8x8 np.ndarray for every character in the string."""

    board = chess.Board()
    initial_states = []
    count = 1

    # Scan 1: Creates states, with length = number of moves in the game
    initial_states.append(custom_board_to_state_fn(board, skill))
    # Apply each move to the board
    for move in moves_string.split():
        try:
            count += 1
            # Skip move numbers
            if "." in move:
                board.push_san(move.split(".")[1])
            else:
                board.push_san(move)

            initial_states.append(custom_board_to_state_fn(board, skill))
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
    skill_array: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Given a list of strings of PGN format moves, create a tensor of shape (len(moves_strings), 8, 8).
    custom_board_to_state is a function that takes a chess.Board object and returns a 8x8 np.ndarray for
    board state, or 1x1 for centipawn advantage."""
    state_stacks = []
    skill = None

    for idx, board in enumerate(moves_strings):
        if skill_array is not None:
            skill = skill_array[idx]
        state_stack = torch.tensor(
            create_state_stack(board, custom_board_to_state_fn, skill)
        ).to(dtype=torch.float32)
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
    user_mapping: Optional[dict[int, int]] = None,
) -> torch.Tensor:
    """Input shape: assert(state_stacks_all_chars.shape) == (modes, sample_size, game_length, rows, cols)
    Output shape: assert(state_stacks_one_hot.shape) == (modes, sample_size, game_length, rows, cols, one_hot_range)
    """
    range_size = max_val - min_val + 1

    mapping = {}
    if user_mapping:
        mapping = user_mapping
        min_val = min(mapping.values())
        max_val = max(mapping.values())
        range_size = max_val - min_val + 1
    else:
        for val in range(min_val, max_val + 1):
            mapping[val] = val - min_val

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

    for val in mapping:
        one_hot[..., mapping[val]] = state_stack == val

    return one_hot


def one_hot_to_state_stack(one_hot: torch.Tensor, min_val: int) -> np.ndarray:
    """Input shape: assert(probe_out.shape) == (modes, sample_size, num_white_moves, rows, cols, one_hot_range)
    Output shape: assert(state_stacks_probe_outputs.shape) == (modes, sample_size, num_white_moves, rows, cols)
    """
    if one_hot.is_cuda:
        one_hot = one_hot.cpu()
    indices = torch.argmax(one_hot, dim=-1)
    state_stack = indices.numpy() + min_val
    return state_stack


def square_to_coordinate(square: chess.Square) -> tuple[int, int]:
    row = chess.square_rank(square)
    column = chess.square_file(square)
    return (row, column)


def find_dots_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every '.' in the string.
    This will hopefully provide a reasonable starting point for training a linear probe.
    """
    indices = [index for index, char in enumerate(moves_string) if char == "."]
    return indices


def find_spaces_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every ' ' in the string."""
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    return indices


def find_odd_spaces_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of odd indices of every ' ' in the string.
    There is some duplicated logic but it simplifies using the Callable function."""
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    # Select only the odd indices: start from index 1, go till the end, step by 2
    odd_indices = indices[1::2]
    return odd_indices


def find_even_spaces_indices(moves_string: str) -> list[int]:
    """Returns a list of ints of even indices of every ' ' in the string.
    There is some duplicated logic but it simplifies using the Callable function."""
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    # Select only the even indices: start from index 0, go till the end, step by 2
    even_indices = indices[::2]
    return even_indices


def find_even_indices_offset_one(moves_string: str) -> list[int]:
    """
    Returns a list of ints of even indices of every ' ' in the string, each incremented by one.
    If the incremented index would be greater than the length of the string, it is not included.
    """
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    even_indices = indices[::2]

    # Increment each even index by one, ensuring it doesn't exceed the string length
    incremented_indices = [
        index + 1 for index in even_indices if index + 1 < len(moves_string)
    ]

    return incremented_indices


def find_odd_indices_offset_one(moves_string: str) -> list[int]:
    """
    Returns a list of ints of odd indices of every ' ' in the string, each incremented by one.
    If the incremented index would be greater than the length of the string, it is not included.
    """
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    odd_indices = indices[1::2]

    # Increment each odd index by one, ensuring it doesn't exceed the string length
    incremented_indices = [
        index + 1 for index in odd_indices if index + 1 < len(moves_string)
    ]

    return incremented_indices


def find_custom_indices(
    df_filename: str, custom_indexing_fn: Callable[[str], list[int]]
) -> np.ndarray:
    df = pd.read_csv(df_filename)
    indices_series = df["transcript"].apply(custom_indexing_fn)
    shortest_length = indices_series.apply(len).min()
    print("Shortest length:", shortest_length)

    indices_series = indices_series.apply(lambda x: x[:shortest_length])
    assert all(
        len(lst) == shortest_length for lst in indices_series
    ), "Not all lists have the same length"

    indices = np.array(indices_series.apply(list).tolist())
    return indices


def encode_string(meta: dict, s: str) -> list[int]:
    """Encode a string into a list of integers."""
    # This is how you get meta
    # with open("meta.pkl", "rb") as f:
    #     meta = pickle.load(f)
    stoi = meta["stoi"]
    return [stoi[c] for c in s]


def decode_list(meta: dict, l: list[int]) -> str:
    """Decode a list of integers into a string."""
    # with open("meta.pkl", "rb") as f:
    #     meta = pickle.load(f)
    itos = meta["itos"]
    return "".join([itos[i] for i in l])


# Adapted from nanogpt
@torch.no_grad()
def get_model_move(
    model,
    meta: dict,
    idx: torch.Tensor,
    max_new_tokens: int = 10,
    temperature=1.0,
    block_size=1023,
):
    """Generate new tokens from a trained language model. If temperature is 0.0, greedy decoding is used.
    Otherwise, standard temperature based sampling is used."""

    if temperature < 0:
        raise ValueError("temperature has to be non-negative")

    input_length = len(idx[0])
    space_idx = encode_string(meta, " ")[0]
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        if temperature == 0.0:
            # greedy decoding
            # model(idx_cond) is a tensor of shape (batch_size, sequence_length, vocab_size)
            # logits is a tensor of shape (batch_size, vocab_size)
            # idx_next is a tensor of shape (batch_size, 1)
            logits = model(idx_cond)[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1).unsqueeze(-1)
        else:
            # forward the model to get the logits for the index in the sequence
            logits = model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next[0] == space_idx:
            break
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    model_response = idx[:, input_length:]
    model_move = decode_list(meta, model_response[0].tolist())
    return model_move
