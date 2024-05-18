import chess
import pandas as pd
import torch
from torch.nn import functional as F
from typing import Callable, Optional
from dataclasses import dataclass
from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
from enum import Enum
import othello_utils

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
BLANK_INDEX = PIECE_TO_ONE_HOT_MAPPING[0]
ONE_HOT_TO_PIECE_MAPPING = {value: key for key, value in PIECE_TO_ONE_HOT_MAPPING.items()}


def board_to_random_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return a 8x8 torch.Tensor.
    Every square should be randomly assigned to 1, -1, or 0.
    This is to sanity check the linear probe.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""
    state_RR = torch.zeros((8, 8), dtype=torch.int)
    for i in range(64):
        state_RR[i // 8, i % 8] = torch.randint(-1, 2, (1,))

    return state_RR


def board_to_skill_state(board: chess.Board, skill: float) -> torch.Tensor:
    """Given a chess board object, return a 1x1 torch.Tensor.
    The 1x1 array should tell what skill level the player is."""
    state_RR = torch.zeros((1, 1), dtype=torch.int)
    state_RR[0][0] = skill

    return state_RR


# import chess.engine

# stockfish_path = "/usr/games/stockfish"
# engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)


def board_to_eval_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return a 1x1 torch.Tensor.
    The 1x1 array should tell which player is winning.
    -1 = Black has > 100 centipawns advantage, 0 = Draw, 1 = White has > 100 centipawns advantage.
    This is horribly inefficient and takes ~0.75 seconds per game. However, I'm just doing exploratory analysis.
    If we wanted efficiency, we could use a bunch of parallel CPU workers to evaluate the board state and store it
    in a lookup table. But, then we couldn't cleanly use this with the existing abstractions.
    To use this function, uncomment the import chess.engine through engine = above, and the internal code below.
    """
    state_RR = torch.zeros((1, 1), dtype=torch.int)

    # info = engine.analyse(board, chess.engine.Limit(time=0.01))
    # score = info["score"].white().score(mate_score=10000)

    # # Modify player_one_score based on the score
    # if score < 100:
    #     state_RR[0][0] = -1
    # elif score > 100:
    #     state_RR[0][0] = 1
    # else:
    #     state_RR[0][0] = 0

    return state_RR


def board_to_piece_color_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return a 8x8 torch.Tensor.
    The 8x8 array should tell if each square is black, white, or blank.
    White is 1, black is -1, and blank is 0.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""
    state_RR = torch.zeros((8, 8), dtype=torch.int)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            # Assign 1 for white pieces and -1 for black pieces
            state_RR[i // 8, i % 8] = 1 if piece.color == chess.WHITE else -1

    return state_RR


def board_to_piece_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return an 8x8 torch.Tensor.
    The 8x8 array should tell what piece is on each square. A white pawn could be 1, a black pawn could be -1, etc.
    Blank squares should be 0.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc."""

    # Because state_RR is initialized to all 0s, we only need to change the values of the pieces
    state_RR = torch.zeros((8, 8), dtype=torch.int)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_value = PIECE_TO_INT[piece.piece_type]
            # Multiply by -1 if the piece is black
            if piece.color == chess.BLACK:
                piece_value *= -1
            state_RR[i // 8, i % 8] = piece_value

    return state_RR


def board_to_pin_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return a 1x1 torch.Tensor.
    The 1x1 array indicates if there are any pins on the board (1 = yes, 0 = no)."""

    state_RR = torch.zeros((1, 1), dtype=torch.int)

    # NOTE: Due to the model's MINE / YOURS / BLANK ontology, we should check for White XOR Black pins
    for color in [chess.WHITE]:
        for i in range(64):
            piece = board.piece_at(i)
            if piece and piece.color == color:
                if board.is_pinned(color, i):
                    state_RR[0, 0] = 1
                    return state_RR

    return state_RR


def board_to_threat_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return an 8x8 torch.Tensor.
    The 8x8 array should tell if each square is being attacked by the opponent."""

    ATTACKING_COLOR = chess.BLACK
    # Because state is initialized to all 0s, we only need to change the values of the pieces
    state_RR = torch.zeros((8, 8), dtype=torch.int)
    for i in range(64):
        if board.is_attacked_by(ATTACKING_COLOR, i):
            state_RR[i // 8, i % 8] = 1

    return state_RR


def board_to_prev_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return an 8x8 torch.Tensor.
    The 8x8 array should tell what piece is on each square at a previous board state."""

    PREVIOUS_TURNS = 25
    state_RR = torch.zeros((8, 8), dtype=torch.int)

    # If we cannot roll back PREVIOUS_TURNS, return a blank state
    # Predicting blank states is trivial, so be careful and change pos_start to not index into the blank states
    if len(board.move_stack) < PREVIOUS_TURNS:
        return state_RR

    new_board = board.copy()

    for _ in range(PREVIOUS_TURNS):
        new_board.pop()

    for i in range(64):
        piece = new_board.piece_at(i)
        if piece:
            piece_value = PIECE_TO_INT[piece.piece_type]
            # Multiply by -1 if the piece is black
            if piece.color == chess.BLACK:
                piece_value *= -1
            state_RR[i // 8, i % 8] = piece_value

    return state_RR


def board_to_legal_moves_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Return an 8x8 torch.Tensor indicating squares where White has legal moves.

    Each square in the array is 1 if White can legally move a piece to that square, otherwise 0.
    In the 8x8 array, row 0 corresponds to A1-H1 (from White's perspective), row 1 to A2-H2, etc.
    """
    MOVING_COLOR = chess.WHITE
    # Initialize the state array with all zeros
    state_RR = torch.zeros((8, 8), dtype=torch.int)

    # Iterate through all legal moves for White
    for move in board.legal_moves:
        # Check if the move is for a White piece
        if board.color_at(move.from_square) == MOVING_COLOR:
            # Update the state_RR array for the destination square of the move
            to_square = move.to_square
            state_RR[to_square // 8, to_square % 8] = 1

    return state_RR


def board_to_last_self_move_state(board: chess.Board, skill: Optional[int] = None) -> torch.Tensor:
    """Given a chess board object, return an 8x8 torch.Tensor.
    All squares will be 0 except for the square where the last white move was made.
    In the 8x8 array, row 0 is A1-H1 (White), row 1 is A2-H2, etc.
    The purpose of this is to see if the linear probe can determine the next move of the GPT.
    To get next move instead of last move, we offset the state stack by 1 in linear_probe_forward_pass():
    resid_post = resid_post[:, :-1, :]
    state_stack_one_hot = state_stack_one_hot[:, :, 1:, :, :, :]
    """

    state_RR = torch.zeros((8, 8), dtype=torch.int)

    # If offset is 2, we are predicting the LLM's next move
    # If offset is 1, we are predicting the opponent's response to the LLM's next move
    offset = 2

    # If there is no last move (such as beginning of game), return the state as is
    if len(board.move_stack) < offset:
        return state_RR

    last_last_move = board.move_stack[-offset]
    destination_square = last_last_move.to_square
    moved_piece = board.piece_at(destination_square)
    if moved_piece is None:
        raise ValueError("Piece type is None")
    piece_value = PIECE_TO_INT[moved_piece.piece_type]
    if moved_piece.color == chess.BLACK:
        piece_value *= -1
    state_RR[destination_square // 8, destination_square % 8] = piece_value

    return state_RR


def state_stack_to_chess_board(state_RR: torch.Tensor) -> chess.Board:
    """Given a state stack, return a chess.Board object.
    WARNING: The board will not include any information about whose turn it is, castling rights, en passant, etc.
    For this reason, pgn_string_to_board is preferred."""
    board = chess.Board(fen=None)
    for row_idx, row in enumerate(state_RR):
        for col_idx, piece in enumerate(row):
            if piece != 0:
                piece_type = abs(piece)
                color = chess.WHITE if piece > 0 else chess.BLACK
                board.set_piece_at(chess.square(col_idx, row_idx), chess.Piece(piece_type, color))
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
    custom_board_to_state_fn: Callable[[chess.Board], torch.Tensor],
    skill: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Given a string of PGN format moves, create an 8x8 torch.Tensor for every character in the string."""

    board = chess.Board()
    initial_states_lRR = []
    count = 1

    # Scan 1: Creates states, with length = number of moves in the game
    initial_states_lRR.append(custom_board_to_state_fn(board, skill).to(dtype=torch.int8))
    # Apply each move to the board
    for move in moves_string.split():
        try:
            count += 1
            # Skip move numbers
            if "." in move:
                board.push_san(move.split(".")[1])
            else:
                board.push_san(move)

            initial_states_lRR.append(custom_board_to_state_fn(board, skill).to(dtype=torch.int8))
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
    expanded_states_lRR = []
    move_index = 0
    for char in moves_string:
        if char == " ":
            move_index += 1
        expanded_states_lRR.append(initial_states_lRR[min(move_index, len(initial_states_lRR) - 1)])

    # expanded_states.append(initial_states[-1]) # The last element in expanded_states is the final position of the board.
    # Currently not using this as len(expanded_states) would be 1 greater than len(moves_string) and that would be confusing.
    return torch.stack(expanded_states_lRR)


def create_state_stacks(
    moves_strings: list[str],
    custom_board_to_state_fn: Callable[[chess.Board], torch.Tensor],
    skill_array: Optional[torch.Tensor] = None,
) -> Float[Tensor, "modes sample_size pgn_str_length rows cols"]:
    """Given a list of strings of PGN format moves, create a tensor of shape (len(moves_strings), 8, 8).
    custom_board_to_state is a function that takes a chess.Board object and returns a 8x8 torch.Tensor for
    board state, or 1x1 for centipawn advantage."""
    state_stacks_BlRR = []
    skill = None

    for idx, pgn_string in enumerate(moves_strings):
        if skill_array is not None:
            skill = skill_array[idx]
        state_stack_lRR = create_state_stack(pgn_string, custom_board_to_state_fn, skill)

        state_stacks_BlRR.append(state_stack_lRR)

    # Convert the list of tensors to a single tensor
    final_state_stack_BlRR = torch.stack(state_stacks_BlRR)
    final_state_stack_MBlRR = final_state_stack_BlRR.unsqueeze(0)  # Add a dimension for the modes
    # Currently, there is just one mode and it isn't necessary. For now, I'm maintaining the dimension for future use.
    return final_state_stack_MBlRR


def state_stack_to_one_hot(
    num_modes: int,
    num_rows: int,
    num_cols: int,
    min_val: int,
    max_val: int,
    device: torch.device,
    state_stack_MBLRR: torch.Tensor,
    user_mapping: Optional[dict[int, int]] = None,
) -> Int[Tensor, "modes sample_size num_white_moves rows cols one_hot_range"]:
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
    one_hot_MBLRRC = torch.zeros(
        state_stack_MBLRR.shape[0],  # num modes
        state_stack_MBLRR.shape[1],  # num games
        state_stack_MBLRR.shape[2],  # num moves
        num_rows,
        num_cols,
        range_size,
        device=device,
        dtype=torch.int8,
    )

    for val in mapping:
        one_hot_MBLRRC[..., mapping[val]] = state_stack_MBLRR == val

    return one_hot_MBLRRC


def one_hot_to_state_stack(one_hot_MBLRRC: torch.Tensor, min_val: int) -> torch.Tensor:
    """We assume input shape UBLRRC, but it could work with other shapes."""
    indices = torch.argmax(one_hot_MBLRRC, dim=-1)
    state_stack_MBLRR = indices + min_val
    return state_stack_MBLRR


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


def get_all_white_pos_indices(moves_string: str) -> list[list[int]]:
    """From this pgn string: ;1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Qxd4 a6 5.Bc4 Nc6 6.Qd1...
    Return a list of lists of indices that correspond to the chars in parentheses:
    (;1.e4)< c5>( 2.Nf3)< d6>( 3.d4)< cxd4>( 4.Qxd4)< a6>( 5.Bc4)< Nc6>( 6.Qd1)"""
    space_indices = find_spaces_indices(moves_string)
    white_move_indices: list[list[int]] = []
    start_index = 0

    if len(space_indices) == 0:
        return [list(range(0, len(moves_string)))]

    for i, space in enumerate(space_indices):
        if i % 2 == 1:
            start_index = space
            if i == len(space_indices) - 1:
                white_move_indices.append(list(range(start_index, len(moves_string))))
                break
            continue
        white_move_indices.append(list(range(start_index, space)))
    return white_move_indices


def get_all_black_pos_indices(moves_string: str) -> list[list[int]]:
    """From this pgn string: ;1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Qxd4 a6 5.Bc4 Nc6 6.Qd1...
    Return a list of lists of indices that correspond to the chars in brackets:
    (;1.e4)< c5>( 2.Nf3)< d6>( 3.d4)< cxd4>( 4.Qxd4)< a6>( 5.Bc4)< Nc6>( 6.Qd1)"""
    space_indices = find_spaces_indices(moves_string)
    black_move_indices: list[list[int]] = []

    if len(space_indices) == 0:
        return []

    start_index = space_indices[0]

    for i, space in enumerate(space_indices):
        if i % 2 == 0:
            start_index = space
            if i == len(space_indices) - 1:
                black_move_indices.append(list(range(start_index, len(moves_string))))
                break
            continue
        black_move_indices.append(list(range(start_index, space)))
    return black_move_indices


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


def find_dots_indices_offset_one(moves_string: str) -> list[int]:
    """Returns a list of ints of indices of every '.' in the string.
    This will hopefully provide a reasonable starting point for training a linear probe.
    """
    indices = [index for index, char in enumerate(moves_string) if char == "."]

    incremented_indices = [index + 1 for index in indices if index + 1 < len(moves_string)]

    return incremented_indices


def find_even_indices_offset_one(moves_string: str) -> list[int]:
    """
    Returns a list of ints of even indices of every ' ' in the string, each incremented by one.
    If the incremented index would be greater than the length of the string, it is not included.
    """
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    even_indices = indices[::2]

    # Increment each even index by one, ensuring it doesn't exceed the string length
    incremented_indices = [index + 1 for index in even_indices if index + 1 < len(moves_string)]

    return incremented_indices


def find_odd_indices_offset_one(moves_string: str) -> list[int]:
    """
    Returns a list of ints of odd indices of every ' ' in the string, each incremented by one.
    If the incremented index would be greater than the length of the string, it is not included.
    """
    indices = [index for index, char in enumerate(moves_string) if char == " "]
    odd_indices = indices[1::2]

    # Increment each odd index by one, ensuring it doesn't exceed the string length
    incremented_indices = [index + 1 for index in odd_indices if index + 1 < len(moves_string)]

    return incremented_indices


def find_custom_indices(custom_indexing_fn: Callable, games_strs_Bl: list) -> torch.Tensor:

    shortest_length = 1e6
    custom_indices = []
    for pgn in games_strs_Bl:
        indices = custom_indexing_fn(pgn)
        shortest_length = min(shortest_length, len(indices))
        custom_indices.append(indices)
    print("Shortest length:", shortest_length)

    for i, indices in enumerate(custom_indices):
        custom_indices[i] = indices[:shortest_length]

    indices = torch.tensor(custom_indices, dtype=torch.int)

    return indices


def encode_string(meta: dict, s: str) -> list[int]:
    """Encode a string into a list of integers."""
    stoi = meta["stoi"]
    return [stoi[c] for c in s]


def decode_list(meta: dict, l: list[int]) -> str:
    """Decode a list of integers into a string."""
    itos = meta["itos"]
    return "".join([itos[i] for i in l])


# Adapted from nanogpt
def get_model_move(
    model,
    meta: dict,
    idx: torch.Tensor,
    max_new_tokens: int = 7,
    temperature=1.0,
    block_size=1023,
):
    """Generate new tokens from a trained language model. If temperature is 0.0, greedy decoding is used.
    Otherwise, standard temperature based sampling is used."""

    if temperature < 0:
        raise ValueError("temperature has to be non-negative")

    input_length = len(idx[0])
    space_idx = encode_string(meta, " ")[0]
    with torch.inference_mode():
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


class PlayerColor(Enum):
    WHITE = "White"
    BLACK = "Black"


@dataclass
class Config:
    min_val: int
    max_val: int
    custom_board_state_function: callable
    linear_probe_name: str
    custom_indexing_function: callable = find_dots_indices
    num_rows: int = 8
    num_cols: int = 8
    levels_of_interest: Optional[list[int]] = None
    column_name: str = None
    probing_for_skill: bool = False
    # pos_start indexes into custom_indexing_function. Example: if pos_start = 25, for find_dots_indices, selects everything after the first 25 moves
    pos_start: int = 0
    # If pos_end is None, it's set to the length of the shortest game in construct_linear_probe_data()
    pos_end: Optional[int] = None
    player_color: PlayerColor = PlayerColor.WHITE
    othello: bool = False


piece_config = Config(
    min_val=-6,
    max_val=6,
    custom_board_state_function=board_to_piece_state,
    linear_probe_name="chess_piece_probe",
)

pin_config = Config(
    min_val=0,
    max_val=1,
    custom_board_state_function=board_to_pin_state,
    num_rows=1,
    num_cols=1,
    linear_probe_name="chess_pin_probe",
)

color_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=board_to_piece_color_state,
    linear_probe_name="chess_color_probe",
)

threat_config = Config(
    min_val=0,
    max_val=1,
    custom_board_state_function=board_to_threat_state,
    linear_probe_name="chess_threat_probe",
)

legal_move_config = Config(
    min_val=0,
    max_val=1,
    custom_board_state_function=board_to_legal_moves_state,
    linear_probe_name="chess_legal_move_probe",
)

prev_move_config = Config(
    min_val=-6,
    max_val=6,
    custom_board_state_function=board_to_prev_state,
    linear_probe_name="chess_prev_move_probe",
    pos_start=15,
    pos_end=16,
)

random_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=board_to_random_state,
    linear_probe_name="chess_random_probe",
)

eval_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=board_to_eval_state,
    linear_probe_name="chess_eval_probe",
    num_rows=1,
    num_cols=1,
)

skill_config = Config(
    min_val=-2,
    max_val=20,
    custom_board_state_function=board_to_skill_state,
    linear_probe_name="chess_skill_probe",
    num_rows=1,
    num_cols=1,
    levels_of_interest=[0, 5],
    probing_for_skill=True,
    pos_start=25,
)

othello_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=othello_utils.games_batch_to_state_stack_mine_yours_BLRRC,
    linear_probe_name="othello_mine_yours_probe",
    othello=True,
)

othello_valid_moves_config = Config(
    min_val=0,
    max_val=1,
    custom_board_state_function=othello_utils.games_batch_to_valid_moves_BLRRC,
    linear_probe_name="othello_valid_moves_probe",
    othello=True,
)


def find_config_by_name(config_name: str) -> Config:
    """
    Finds and returns the Config instance with a matching linear_probe_name.
    """
    all_configs = [piece_config, color_config, random_config, skill_config, othello_config]
    for config in all_configs:
        if config.linear_probe_name == config_name:
            return config
    raise ValueError(f"Config with name {config_name} not found")


def update_config_using_player_color(
    player_color: PlayerColor, config: Config, custom_function: Optional[Callable] = None
) -> Config:
    """Player color will determine which indexing function we use. In addition, we set player to white by default.
    If player is black, then we update the probe name as well."""

    if custom_function:
        config.custom_indexing_function = custom_function
        config.player_color = player_color
        return config

    if player_color == PlayerColor.WHITE:
        config.custom_indexing_function = find_dots_indices
        config.player_color = player_color

    if player_color == PlayerColor.BLACK:
        config.linear_probe_name = config.linear_probe_name.replace("probe", "black_player_probe")
        config.custom_indexing_function = find_even_spaces_indices
        config.player_color = player_color

    return config


def set_config_min_max_vals_and_column_name(
    config: Config,
    input_dataframe_file: str,
    dataset_prefix: str,
) -> Config:
    if config.levels_of_interest is not None or config.probing_for_skill:
        if dataset_prefix == "stockfish_":
            config.column_name = "player_two"
        elif "lichess_" in dataset_prefix:
            config.column_name = "WhiteEloBinIndex"
    else:
        return config
    df = pd.read_csv(input_dataframe_file)
    config.min_val = df[config.column_name].min()
    config.max_val = df[config.column_name].max()

    return config
