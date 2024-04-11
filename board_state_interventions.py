import torch
from fancy_einsum import einsum
import chess
from dataclasses import dataclass, field, fields
import pickle
import logging
from functools import partial
from enum import Enum
import json

import chess_utils
import train_test_chess

from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
from beartype import beartype

import cProfile
import pstats
import io

torch.set_grad_enabled(False)

# Flags to control logging
debug_mode = False
info_mode = True

if debug_mode:
    log_level = logging.DEBUG
elif info_mode:
    log_level = logging.INFO
else:
    log_level = logging.WARNING

# Configure logging
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

GPT_LAYER_COUNT = 8
DATA_DIR = "data/"
SAVED_PROBE_DIR = f"linear_probes/saved_probes/"
RECORDING_DIR = "intervention_logs/"
SPLIT = "test"
MODES = 1  # Currently only supporting 1 mode so this is fairly unnecessary
START_POS = 0
END_POS = 30
BLANK_INDEX = chess_utils.PIECE_TO_ONE_HOT_MAPPING[0]
SAMPLING_MOVES = 5
TEMPERATURE = 1.0
MAX_GAMES = 5000

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
logger.info(f"Using device: {DEVICE}")

with open("models/meta.pkl", "rb") as f:
    META = pickle.load(f)


class InterventionType(Enum):
    SINGLE_SCALE = "single_scale"
    SINGLE_TARGET = "single_target"
    AVERAGE_TARGET = "average_target"


class ModelType(Enum):
    ORIGINAL = "original"
    MODIFIED = "modified"


@dataclass
class MoveTracker:
    orig_board_argmax_legal_total: int = 0
    orig_board_sampled_legal_total: int = 0
    orig_board_sampled_legal_unique: int = 0
    mod_board_argmax_legal_total: int = 0
    mod_board_sampled_legal_total: int = 0
    mod_board_sampled_legal_unique: int = 0
    unique_moves: int = 0

    def update_with(self, other: "MoveTracker"):
        """Updates the attributes of this MoveTracker with values from another."""
        self.orig_board_argmax_legal_total += other.orig_board_argmax_legal_total
        self.orig_board_sampled_legal_total += other.orig_board_sampled_legal_total
        self.orig_board_sampled_legal_unique += other.orig_board_sampled_legal_unique
        self.mod_board_argmax_legal_total += other.mod_board_argmax_legal_total
        self.mod_board_sampled_legal_total += other.mod_board_sampled_legal_total
        self.mod_board_sampled_legal_unique += other.mod_board_sampled_legal_unique
        self.unique_moves += other.unique_moves


@dataclass
class MoveCounters:
    total_moves: int = 0
    possible_moves: int = 0
    orig_model_tracker: MoveTracker = field(default_factory=MoveTracker)
    mod_model_tracker: MoveTracker = field(default_factory=MoveTracker)


def get_probe_data(probe_name: str, num_games: int) -> train_test_chess.LinearProbeData:
    probe_file_location = f"{SAVED_PROBE_DIR}{probe_name}"
    with open(probe_file_location, "rb") as f:
        state_dict = torch.load(f, map_location=torch.device(DEVICE))
        print(state_dict.keys())
        for key in state_dict.keys():
            if key != "linear_probe":
                print(key, state_dict[key])

        config = chess_utils.find_config_by_name(state_dict["config_name"])
        layer = state_dict["layer"]
        model_name = state_dict["model_name"]
        dataset_prefix = state_dict["dataset_prefix"]
        config.pos_start = state_dict["pos_start"]
        levels_of_interest = None
        if "levels_of_interest" in state_dict.keys():
            levels_of_interest = state_dict["levels_of_interest"]
        config.levels_of_interest = levels_of_interest
        n_layers = state_dict["n_layers"]

        split = SPLIT
        input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
        config = chess_utils.set_config_min_max_vals_and_column_name(
            config, input_dataframe_file, dataset_prefix
        )

        probe_data = train_test_chess.construct_linear_probe_data(
            input_dataframe_file,
            dataset_prefix,
            n_layers,
            model_name,
            config,
            num_games,
            DEVICE,
        )
        return probe_data


@jaxtyped(typechecker=beartype)
def prepare_intervention_data(
    probe_names: dict[int, str],
    probe_data: train_test_chess.LinearProbeData,
    num_games: int,
) -> tuple[
    dict[int, Float[Tensor, "modes d_model rows cols options"]],
    Int[Tensor, "modes num_games pgn_str_length rows cols"],
    Int[Tensor, "num_games num_white_moves"],
]:

    probes = {}
    checkpoint = None  # Going to retain the last checkpoint for the config

    for layer, probe_name in probe_names.items():
        probe_file_location = f"{SAVED_PROBE_DIR}{probe_name}"
        checkpoint = torch.load(probe_file_location, map_location=torch.device(DEVICE))
        linear_probe = checkpoint["linear_probe"]
        probes[layer] = linear_probe

    config = chess_utils.find_config_by_name(checkpoint["config_name"])
    state_stacks_all_chars = chess_utils.create_state_stacks(
        probe_data.board_seqs_string[:num_games], config.custom_board_state_function
    )
    logger.info(f"state_stack shape: {state_stacks_all_chars.shape}")
    pgn_str_length = len(probe_data.board_seqs_string[0])

    assert (state_stacks_all_chars.shape) == (
        MODES,
        num_games,
        pgn_str_length,
        config.num_rows,
        config.num_cols,
    )

    white_move_indices = probe_data.custom_indices[:num_games]
    num_white_moves = white_move_indices.shape[1]
    assert (white_move_indices.shape) == (num_games, num_white_moves)

    return probes, state_stacks_all_chars, white_move_indices


def initialize_output_tracker(probes: dict[int, str]) -> dict:
    output_tracker: dict[int, dict] = {}
    for layer in probes:
        output_tracker[layer] = {
            "original_blank_probe": [],
            "modified_blank_probe": [],
            "original_piece_probe": [],
            "modified_piece_probe": [],
            "original_blank_grid": [],
            "modified_blank_grid": [],
            "original_piece_grid": [],
            "modified_piece_grid": [],
            "average_original_blank_grid": [],
            "average_modified_blank_grid": [],
            "average_original_piece_grid": [],
            "average_modified_piece_grid": [],
            "scales": [],
            "successes": [],
            "cells": [],
            "pieces": [],
            "modified_move": [],
            "original_move": [],
        }
    return output_tracker


def initialize_scale_tracker(scales: list[float]) -> dict[float, MoveTracker]:
    scale_tracker: dict[float, MoveTracker] = {}
    for scale in scales:
        scale_tracker[scale] = MoveTracker()
    return scale_tracker


@jaxtyped(typechecker=beartype)
def update_output_tracker_grids(
    probes: dict[int, Float[Tensor, "modes d_model rows cols options"]],
    probe_data: train_test_chess.LinearProbeData,
    model_input: Int[Tensor, "num_games pgn_str_length"],
    state_stacks_all_chars: Int[Tensor, "modes num_games pgn_str_length rows cols"],
    output_tracker: dict,
    move_of_interest_index: int,
    sample_index: int,
    r: int,
    c: int,
    moved_piece_probe_index: int,
    moved_piece_int: int,
    model_move: str,
    model_type: ModelType,
) -> dict:

    model_type_str = model_type.value

    _, cache = probe_data.model.run_with_cache(model_input)

    for layer in output_tracker:
        probe_outputs = calculate_probe_outputs(probes, cache)
        probe_out = probe_outputs[layer]

        blank_probe_grid = probe_out[0, 0, move_of_interest_index, :, :, BLANK_INDEX]
        piece_probe_grid = probe_out[0, 0, move_of_interest_index, :, :, moved_piece_probe_index]

        blank_probe_out = blank_probe_grid[r, c]
        piece_probe_out = piece_probe_grid[r, c]
        output_tracker[layer][f"{model_type_str}_blank_grid"].append(blank_probe_grid.to("cpu"))
        output_tracker[layer][f"{model_type_str}_piece_grid"].append(piece_probe_grid.to("cpu"))
        output_tracker[layer][f"{model_type_str}_blank_probe"].append(blank_probe_out)
        output_tracker[layer][f"{model_type_str}_piece_probe"].append(piece_probe_out)
        output_tracker[layer][f"{model_type_str}_move"].append(model_move)

        average_blank_values = average_probe_empty_cell_value(
            state_stacks_all_chars,
            probe_outputs,
            BLANK_INDEX,
            move_of_interest_index,
            sample_index,
        )
        average_piece_values = average_probe_empty_cell_value(
            state_stacks_all_chars,
            probe_outputs,
            moved_piece_probe_index,
            move_of_interest_index,
            sample_index,
        )

        output_tracker[layer][f"average_{model_type_str}_blank_grid"].append(
            average_blank_values[layer]
        )
        output_tracker[layer][f"average_{model_type_str}_piece_grid"].append(
            average_piece_values[layer]
        )

    if model_type == ModelType.MODIFIED:
        for layer in output_tracker:
            # Duplicating some metadata for each layer but it's a small amount
            output_tracker[layer]["scales"].append(scale)
            output_tracker[layer]["successes"].append(False)
            output_tracker[layer]["cells"].append((r, c))
            output_tracker[layer]["pieces"].append(moved_piece_int)

    return output_tracker


def create_recording_data(
    move_counters: MoveCounters, scale_tracker: dict[float, MoveTracker]
) -> dict:
    records = {}
    records["orig_model_tracker"] = {}
    records["mod_model_tracker"] = {}
    for field in fields(move_counters.orig_model_tracker):
        records["orig_model_tracker"][field.name] = getattr(
            move_counters.orig_model_tracker, field.name
        )
    for field in fields(move_counters.mod_model_tracker):
        records["mod_model_tracker"][field.name] = getattr(
            move_counters.mod_model_tracker, field.name
        )
    for field in fields(move_counters):
        if field.name == "orig_model_tracker" or field.name == "mod_model_tracker":
            continue
        records[field.name] = getattr(move_counters, field.name)
    for scale in scale_tracker:
        records[scale] = {}
        for field in fields(scale_tracker[scale]):
            records[scale][field.name] = getattr(scale_tracker[scale], field.name)
    records["possible_sampled_moves"] = records["possible_moves"] * SAMPLING_MOVES
    return records


def update_move_counters_best_per_move(
    move_counters: MoveCounters,
    per_move_scale_tracker: dict[float, MoveTracker],
) -> MoveCounters:
    """For each move, we find the best performing scale parameter. We then increment the move counter trackers with these values.
    The purpose is to get an upper bound on effectiveness if we could dynamically select a good scale parameter.
    """
    scales = list(per_move_scale_tracker.keys())
    for field in fields(per_move_scale_tracker[scales[0]]):
        best_scale_value = max(
            getattr(per_move_scale_tracker[scale], field.name) for scale in scales
        )
        current_scale_value = getattr(move_counters.mod_model_tracker, field.name)
        setattr(
            move_counters.mod_model_tracker,
            field.name,
            best_scale_value + current_scale_value,
        )

    return move_counters


def sample_moves_from_model(
    model,
    model_input: Int[Tensor, "num_games pgn_str_length"],
    original_board: chess.Board,
    modified_board: chess.Board,
) -> MoveTracker:
    """Samples moves from a model and updates the provided list of boards with the
    total number of legal moves and unique legal moves for each board."""
    unique_moves = set()
    move_tracker = MoveTracker()
    for _ in range(SAMPLING_MOVES):
        sampled_model_move = chess_utils.get_model_move(
            model, META, model_input, temperature=TEMPERATURE
        )
        try:
            original_board.parse_san(sampled_model_move)
            # print(f"Model original move: {sampled_model_move}")
            move_tracker.orig_board_sampled_legal_total += 1
            if sampled_model_move not in unique_moves:
                move_tracker.orig_board_sampled_legal_unique += 1
        except:
            # print(f"Invalid original move: {sampled_model_move}")
            pass
        try:
            modified_board.parse_san(sampled_model_move)
            print(f"Model modified move: {sampled_model_move}")
            move_tracker.mod_board_sampled_legal_total += 1
            if sampled_model_move not in unique_moves:
                move_tracker.mod_board_sampled_legal_unique += 1
        except:
            print(f"Invalid modified move: {sampled_model_move}")
            pass
        unique_moves.add(sampled_model_move)

    move_tracker.unique_moves += len(unique_moves)

    return move_tracker


def check_if_legal_move(board: chess.Board, move: str) -> bool:
    try:
        board.parse_san(move)
        return True
    except:
        return False


@jaxtyped(typechecker=beartype)
def calculate_probe_outputs(
    probes: dict[int, Float[Tensor, "modes d_model rows cols options"]], cache
) -> dict[int, Float[Tensor, "modes batch num_white_moves rows cols options"]]:
    probe_outputs = {}
    for layer in probes:
        resid_post = cache["resid_post", layer][:, :]  # shape is (batch, pos, d_model)
        linear_probe = probes[layer]
        probe_outputs[layer] = einsum(
            "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
            resid_post,
            linear_probe,
        )
    return probe_outputs


@jaxtyped(typechecker=beartype)
def calculate_scale_coefficient(
    model_activations: Float[Tensor, "d_model"],
    flip_dir: Float[Tensor, "d_model"],
    probe: Float[Tensor, "d_model"],
    target: float,
) -> Tensor:
    """Find the scale coefficient that will result in the linear probe output being equal to the target value."""
    left_side = torch.dot(model_activations, probe) - target
    right_side = torch.dot(flip_dir, probe)
    scale = left_side / right_side
    return scale


def get_average_outputs(output_tracker: dict) -> tuple[float, float]:
    sum_first_elements = sum(item[0].item() for item in output_tracker)
    sum_second_elements = sum(item[1].item() for item in output_tracker)
    average_first = sum_first_elements / len(output_tracker)
    average_second = sum_second_elements / len(output_tracker)
    return average_first, average_second


def average_probe_empty_cell_value(
    state_stacks: torch.Tensor,
    probe_outputs: dict[int, torch.Tensor],
    piece_index: int,
    move_of_interest_index,
    sample_index: int,
) -> dict[int, float]:
    """
    If move_of_interest_index is a queen:
    For an 8 x 8 board with 1 queen, find the average probe output pre softmax for all cells that are not a queen.

    Returns:
    - float: The average of the specified values in the tensor. Returns 0 if no relevant values.
    """
    average_cell_values = {}
    for layer in probe_outputs:
        probe_output = probe_outputs[layer]
        target_val = chess_utils.ONE_HOT_TO_PIECE_MAPPING[piece_index]
        probe_state = probe_output[0, 0, move_of_interest_index, :, :, piece_index]
        value_mask = state_stacks[0, sample_index, move_of_interest_index, :, :] != target_val
        value_mask = value_mask.to(DEVICE)

        # Select the relevant values based on the mask
        relevant_values = torch.masked_select(probe_state, value_mask)

        # Compute the mean of relevant values if there are any, otherwise return 0
        if relevant_values.nelement() == 0:
            average_cell_values[layer] = 0.0
        else:
            average_cell_values[layer] = relevant_values.mean().item()
    return average_cell_values


# This is a 250 line function, which I'm not thrilled about. However, every sequential step is only used once in this function.
# I made an initial attempt to break it up into smaller functions, but I found that it made the code harder to follow.
# I also have limited time to refactor this function, so I'm leaving it as is for now.
# There is a lot going on here, but it's all necessary.
def perform_board_interventions(
    probe_names: dict[int, str],
    probe_data: train_test_chess.LinearProbeData,
    num_games: int,
    intervention_type: InterventionType,
    recording_name: str,
    piece_coefficient: float = 1.0,
    blank_coefficient: float = 1.0,
    track_outputs: bool = False,
    scales: list[float] = [0.1],
) -> float:
    probes, state_stacks_all_chars, white_move_indices = prepare_intervention_data(
        probe_names, probe_data, num_games
    )
    # probes is a dict of [int: torch.Tensor]
    # probe is a tensor of shape (modes, d_model, rows, cols, options)
    # state_stacks_all_chars is a tensor of shape (modes, num_games, pgn_str_length, rows, cols)
    # white_move_indices is a tensor of shape (num_games, num_white_moves)
    scale_tracker = initialize_scale_tracker(scales)
    move_counters = MoveCounters()

    # Output tracker stores metadata and the original and modified probe outputs for the entire board per move per game for each layer
    # The results can be viewed as heatmaps per probe output in probe_output_data_exploration.ipynb
    # CAUTION: This can quickly grow to gigabytes of data
    if track_outputs:
        output_tracker = initialize_output_tracker(probes)

    average_piece_values = {}

    for sample_index in range(num_games):
        for scale in scales:
            print(
                f"Scale: {scale}, deterministic count: {scale_tracker[scale].mod_board_argmax_legal_total}, sampled count: {scale_tracker[scale].mod_board_sampled_legal_total}"
            )

        for move_of_interest in range(START_POS, END_POS):
            print(
                f"Sample index: {sample_index}, total moves: {move_counters.total_moves}, possible moves: {move_counters.possible_moves}, legal intervention moves: {move_counters.mod_model_tracker.mod_board_argmax_legal_total}"
            )
            move_counters.total_moves += 1

            # Step 1: Get the board state at move_of_interest
            move_of_interest_index = white_move_indices[sample_index][move_of_interest]
            pgn_string = probe_data.board_seqs_string[sample_index][: move_of_interest_index + 1]
            orig_board = chess_utils.pgn_string_to_board(pgn_string)

            # Step 2: Get the model move at move_of_interest
            # model_input.shape is (1, move_of_interest_index + 1)
            encoded_input = chess_utils.encode_string(META, pgn_string)
            # model input shape: (1, pgn_str_length)
            model_input = torch.tensor(encoded_input).unsqueeze(0).to(DEVICE)
            argmax_model_move = chess_utils.get_model_move(
                probe_data.model, META, model_input, temperature=0.0
            )

            # Step 3: Check if the model move is legal. parse_san will throw an exception if the move is illegal
            try:
                model_move_san = orig_board.parse_san(argmax_model_move)
            except:
                continue

            move_counters.orig_model_tracker.orig_board_argmax_legal_total += 1

            print(f"\nargmax_model_move: {argmax_model_move}\n")

            # Step 4: Determine which piece was moved from which source square
            moved_piece = orig_board.piece_at(model_move_san.from_square)
            if moved_piece is None:
                raise Exception("No piece found at source square")
            moved_piece_int = chess_utils.PIECE_TO_INT[moved_piece.piece_type]
            moved_piece_probe_index = chess_utils.PIECE_TO_ONE_HOT_MAPPING[moved_piece_int]
            r, c = chess_utils.square_to_coordinate(model_move_san.from_square)

            # If the piece is a king, we skip the intervention as a legal chess game must have a king.
            if moved_piece.piece_type == chess.KING:
                continue

            # Step 5: Make a modified board where source square is now empty. Verify that it has legal moves available
            modified_board = orig_board.copy()
            modified_board.set_piece_at(model_move_san.from_square, None)

            if not any(orig_board.legal_moves):
                print("No legal moves available for the modified board. Skipping...")
                continue

            move_counters.possible_moves += 1

            # Step 5.1: Sample n moves from the unmodified model
            # Track how many moves were legal on both the original and modified boards
            move_tracker = sample_moves_from_model(
                probe_data.model, model_input, orig_board, modified_board
            )
            move_counters.orig_model_tracker.update_with(move_tracker)

            # If we are targetting probe output values, collect the average probe output values.
            if intervention_type == InterventionType.AVERAGE_TARGET:
                _, cache = probe_data.model.run_with_cache(model_input)
                probe_outputs = calculate_probe_outputs(probes, cache)
                average_piece_values = average_probe_empty_cell_value(
                    state_stacks_all_chars,
                    probe_outputs,
                    moved_piece_probe_index,
                    move_of_interest_index,
                    sample_index,
                )

            # Initialize some legal move trackers. Note that these get reset every move. Add the end of the move,
            # we find the maximum value of these trackers to get the maximum possible legal moves for each move
            per_move_scale_tracker = initialize_scale_tracker(scales)
            for scale in scales:
                print(f"Scale: {scale}")

                if track_outputs:
                    output_tracker = update_output_tracker_grids(
                        probes,
                        probe_data,
                        model_input,
                        state_stacks_all_chars,
                        output_tracker,
                        move_of_interest_index,
                        sample_index,
                        r,
                        c,
                        moved_piece_probe_index,
                        moved_piece_int,
                        argmax_model_move,
                        ModelType.ORIGINAL,
                    )

                # This is the intervention function. In it, we obtain a vector to flip the square to blank in the model's activations at a given layer
                # Multiply it by some scale factor, then subtract it from the model's activations
                # If we make this function more modular and pass all variables in (probes, r, c, etc), it is much slower
                def flip_hook(
                    resid,  # shape is (1, num_white_moves, d_model)
                    hook,
                    layer: int,
                    scale: float = 0.1,
                ):
                    target = 0.0
                    blank_probe = probes[layer][:, :, r, c, BLANK_INDEX].squeeze()
                    piece_probe = probes[layer][:, :, r, c, moved_piece_probe_index].squeeze()

                    flip_dir = (piece_probe * piece_coefficient) - (blank_probe * blank_coefficient)
                    flip_dir = flip_dir / flip_dir.norm()

                    if (
                        intervention_type == InterventionType.AVERAGE_TARGET
                        or intervention_type == InterventionType.SINGLE_TARGET
                    ):
                        if intervention_type == InterventionType.AVERAGE_TARGET:
                            target = average_piece_values[layer] + scale
                        else:
                            target = scale
                        scale = calculate_scale_coefficient(
                            resid[0, move_of_interest_index, :],
                            flip_dir,
                            piece_probe,
                            float(target),
                        )
                        # scale = min(0.3, scale)
                        # print(target, scale)

                    resid[0, :] -= scale * flip_dir

                    # For experimentation with dynamic scale setting
                    # coeff = resid[0, move_of_interest_index] @ flip_dir / flip_dir.norm()

                    # So we only print once during inference
                    # if resid.shape[1] <= move_of_interest_index + 1:
                    #     print(
                    #         f"Layer: {layer}, coeff: {coeff:10.3f}, scale: {scale:10.3f}, target: {target:10.3f}"
                    #     )

                # Step 6: Intervene on the model's activations and get the model move under the modified board state
                probe_data.model.reset_hooks()
                for layer in probes:
                    temp_hook_fn = partial(flip_hook, layer=layer, scale=scale)
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    probe_data.model.add_hook(hook_name, temp_hook_fn)

                modified_board_argmax_model_move = chess_utils.get_model_move(
                    probe_data.model, META, model_input, temperature=0.0
                )

                print(f"\nModified board argmax model move: {modified_board_argmax_model_move}\n")

                # Step 6.1: Sample n moves from the modified model
                # Track how many moves were legal on the modified board
                # Note that we are tracking this for each scale
                move_tracker = sample_moves_from_model(
                    probe_data.model, model_input, orig_board, modified_board
                )
                per_move_scale_tracker[scale].update_with(move_tracker)

                # Step 6.2: If we are tracking outputs, update the output tracker with the modified outputs
                if track_outputs:
                    output_tracker = update_output_tracker_grids(
                        probes,
                        probe_data,
                        model_input,
                        state_stacks_all_chars,
                        output_tracker,
                        move_of_interest_index,
                        sample_index,
                        r,
                        c,
                        moved_piece_probe_index,
                        moved_piece_int,
                        argmax_model_move,
                        ModelType.MODIFIED,
                    )

                probe_data.model.reset_hooks()

                if check_if_legal_move(modified_board, modified_board_argmax_model_move):
                    # Step 8: The move is legal. Update the legal move trackers
                    if track_outputs:
                        for layer in output_tracker:
                            output_tracker[layer]["successes"][-1] = True
                    per_move_scale_tracker[scale].mod_board_argmax_legal_total += 1

                if check_if_legal_move(orig_board, modified_board_argmax_model_move):
                    per_move_scale_tracker[scale].orig_board_argmax_legal_total += 1

                scale_tracker[scale].update_with(per_move_scale_tracker[scale])

            # Update move_counters with best result per move at end of turn
            move_counters = update_move_counters_best_per_move(
                move_counters, per_move_scale_tracker
            )
        if move_counters.possible_moves > MAX_GAMES:
            break

    # After intervening on all moves in all games, save output_tracker and move_counters to disk
    if track_outputs:
        file_path = "output_tracker.pkl"
        with open(file_path, "wb") as file:
            pickle.dump(output_tracker, file)
        print(f"File saved to {file_path}")
    print(
        f"Sample index: {sample_index}, total moves: {move_counters.total_moves}, possible moves: {move_counters.possible_moves}, legal intervention moves: {move_counters.mod_model_tracker.mod_board_argmax_legal_total}"
    )
    for scale in scales:
        print(
            f"Scale: {scale}, deterministic count: {scale_tracker[scale].mod_board_argmax_legal_total}, sampled count: {scale_tracker[scale].mod_board_sampled_legal_total}"
        )
    recording_name = RECORDING_DIR + "/" + recording_name + ".json"
    with open(recording_name, "w") as file:
        records = create_recording_data(move_counters, scale_tracker)
        file.write(json.dumps(records))

    return (
        move_counters.mod_model_tracker.mod_board_argmax_legal_total / move_counters.possible_moves
    )


if __name__ == "__main__":

    scales_lookup: dict[InterventionType, list[float]] = {
        InterventionType.SINGLE_SCALE: [1.5],
        InterventionType.AVERAGE_TARGET: [9.0],
        InterventionType.SINGLE_TARGET: [-9],
    }

    intervention_types = [
        InterventionType.SINGLE_SCALE,
    ]

    num_games = 200

    for intervention_type in intervention_types:

        probe_names = {}
        first_layer = 5
        last_layer = 5

        for i in range(first_layer, last_layer + 1, 1):
            probe_names[i] = (
                f"tf_lens_lichess_{GPT_LAYER_COUNT}layers_ckpt_no_optimizer_chess_piece_probe_layer_{i}.pth"
            )
        probe_data = get_probe_data(probe_names[first_layer], num_games)

        piece_coe = 1.0
        blank_coe = 0.0

        scales = scales_lookup[intervention_type]

        recording_name = f"n_layers={GPT_LAYER_COUNT}_intervention_type={intervention_type.value}_first_layer={first_layer}_last_layer={last_layer}_p={piece_coe}_b={blank_coe}_scales="
        for scale in scales:
            recording_name += f"{str(scale).replace('.', '')[:5]}_"

        print(f"Recording name: {recording_name}")

        perform_board_interventions(
            probe_names,
            probe_data,
            num_games,
            intervention_type,
            recording_name,
            track_outputs=False,
            scales=scales,
        )

# For profiling, most cumulative time appears to be in forward pass in chess_utils.get_model_move()
# def run_profile():
#     pr = cProfile.Profile()
#     pr.enable()

#     perform_board_interventions(
#         probe_names,
#         probe_data,
#         1,
#         intervention_type,
#         recording_name,
#         track_outputs=False,
#         scales=scales,
#     )

#     pr.disable()
#     s = io.StringIO()
#     ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
#     ps.print_stats()
#     print(s.getvalue())


# run_profile()
