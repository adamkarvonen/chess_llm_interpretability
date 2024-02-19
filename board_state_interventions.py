import torch
import numpy as np
from fancy_einsum import einsum
import chess
import numpy as np
from dataclasses import dataclass, fields
import pickle
import logging
from functools import partial
from enum import Enum
import json

import chess_utils
import train_test_chess

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

MODEL_DIR = "models/"
DATA_DIR = "data/"
PROBE_DIR = "linear_probes/"
SAVED_PROBE_DIR = "linear_probes/8layer_piece_probe_sweep/"
RECORDING_DIR = "intervention_logs/"
SPLIT = "test"
MODES = 1  # Currently only supporting 1 mode so this is fairly unnecessary
START_POS = 5
END_POS = 30
BLANK_INDEX = chess_utils.PIECE_TO_ONE_HOT_MAPPING[0]
SAMPLING_MOVES = 10
TEMPERATURE = 1.0
GPT_LAYER_COUNT = 8

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
logger.info(f"Using device: {device}")

with open("models/meta.pkl", "rb") as f:
    META = pickle.load(f)


class InterventionType(Enum):
    SINGLE_SCALE = "single_scale"
    SINGLE_TARGET = "single_target"
    AVERAGE_TARGET = "average_target"


class SamplingType(Enum):
    DETERMINISTIC_ONLY = "deterministic_only"
    # BOTH means determinstic sampling and probabilistic temperature based sampling
    BOTH = "both"


@dataclass
class MoveCounters:
    total_moves: int = 0
    possible_moves: int = 0
    original_board_deterministic_legal_moves: int = 0
    original_model_original_board_sampled_legal_moves: int = 0
    original_model_modified_board_sampled_legal_moves: int = 0
    modified_board_deterministic_legal_moves: int = 0
    modified_model_modified_board_sampled_legal_moves: int = 0


def get_probe_data(probe_name: str) -> train_test_chess.LinearProbeData:
    probe_file_location = f"{SAVED_PROBE_DIR}{probe_name}"
    with open(probe_file_location, "rb") as f:
        state_dict = torch.load(f, map_location=torch.device(device))
        print(state_dict.keys())
        for key in state_dict.keys():
            if key != "linear_probe":
                print(key, state_dict[key])

        config = train_test_chess.find_config_by_name(state_dict["config_name"])
        layer = state_dict["layer"]
        model_name = state_dict["model_name"]
        dataset_prefix = state_dict["dataset_prefix"]
        process_data = state_dict["process_data"]
        column_name = state_dict["column_name"]
        config.pos_start = state_dict["pos_start"]
        levels_of_interest = None
        if "levels_of_interest" in state_dict.keys():
            levels_of_interest = state_dict["levels_of_interest"]
        config.levels_of_interest = levels_of_interest
        indexing_function_name = state_dict["indexing_function_name"]
        n_layers = state_dict["n_layers"]

        split = SPLIT
        input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
        config = train_test_chess.set_config_min_max_vals_and_column_name(
            config, input_dataframe_file, dataset_prefix
        )

        probe_data = train_test_chess.construct_linear_probe_data(
            input_dataframe_file,
            layer,
            dataset_prefix,
            split,
            n_layers,
            model_name,
            config,
        )
        return probe_data


def prepare_intervention_data(
    probe_names: dict[int, str],
    probe_data: train_test_chess.LinearProbeData,
    sample_size: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:

    probes = {}
    checkpoint = None  # Going to retain the last checkpoint for the config

    for layer, probe_name in probe_names.items():
        probe_file_location = f"{SAVED_PROBE_DIR}{probe_name}"
        checkpoint = torch.load(probe_file_location, map_location=torch.device(device))
        linear_probe = checkpoint["linear_probe"]
        probes[layer] = linear_probe

    config = train_test_chess.find_config_by_name(checkpoint["config_name"])
    state_stacks_all_chars = chess_utils.create_state_stacks(
        probe_data.board_seqs_string[:sample_size], config.custom_board_state_function
    )
    logger.info(f"state_stack shape: {state_stacks_all_chars.shape}")
    game_length_in_chars = len(probe_data.board_seqs_string[0])

    assert (state_stacks_all_chars.shape) == (
        MODES,
        sample_size,
        game_length_in_chars,
        config.num_rows,
        config.num_cols,
    )

    white_move_indices = probe_data.custom_indices[:sample_size]
    num_white_moves = white_move_indices.shape[1]
    assert (white_move_indices.shape) == (sample_size, num_white_moves)

    return probes, state_stacks_all_chars, white_move_indices


def initialize_output_tracker(probes: dict[int, str]) -> dict:
    output_tracker = {}
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


def update_output_tracker_grids(
    probe_outputs: dict[int, torch.Tensor],
    output_tracker: dict,
    layer: int,
    move_of_interest_index: int,
    r: int,
    c: int,
    moved_piece_probe_index: int,
    model_move: str,
    original_or_modified: str,
) -> dict:

    if original_or_modified != "original" and original_or_modified != "modified":
        raise ValueError("original_or_modified must be either 'original' or 'modified'")

    for layer in probe_outputs:
        probe_out = probe_outputs[layer]

        blank_probe_grid = probe_out[0, 0, move_of_interest_index, :, :, BLANK_INDEX]
        piece_probe_grid = probe_out[
            0, 0, move_of_interest_index, :, :, moved_piece_probe_index
        ]

        blank_probe_out = blank_probe_grid[r, c]
        piece_probe_out = piece_probe_grid[r, c]
        output_tracker[layer][f"{original_or_modified}_blank_grid"].append(
            blank_probe_grid.to("cpu")
        )
        output_tracker[layer][f"{original_or_modified}_piece_grid"].append(
            piece_probe_grid.to("cpu")
        )
        output_tracker[layer][f"{original_or_modified}_blank_probe"].append(
            blank_probe_out
        )
        output_tracker[layer][f"{original_or_modified}_piece_probe"].append(
            piece_probe_out
        )
        output_tracker[layer][f"{original_or_modified}_move"].append(model_move)

    return output_tracker


def calculate_probe_outputs(
    probes: dict[int, torch.Tensor], cache
) -> dict[int, torch.Tensor]:
    probe_outputs = {}
    for layer in probes:
        resid_post = cache["resid_post", layer][:, :]
        linear_probe = probes[layer]
        probe_outputs[layer] = einsum(
            "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
            resid_post,
            linear_probe,
        )
    return probe_outputs


def calculate_scale_coefficient(
    model_activations: torch.Tensor,
    flip_dir: torch.Tensor,
    probe: torch.Tensor,
    target: float,
) -> float:
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
        value_mask = (
            state_stacks[0, sample_index, move_of_interest_index, :, :] != target_val
        )
        value_mask = value_mask.to(device)

        # Select the relevant values based on the mask
        relevant_values = torch.masked_select(probe_state, value_mask)

        # Compute the mean of relevant values if there are any, otherwise return 0
        if relevant_values.nelement() == 0:
            average_cell_values[layer] = 0
        else:
            average_cell_values[layer] = relevant_values.mean().item()
    return average_cell_values


# This is a 300 line function, which I'm not thrilled about. However, every sequential step is only used once in this function.
# I made an initial attempt to break it up into smaller functions, but I found that it made the code harder to follow.
# I also have limited time to refactor this function, so I'm leaving it as is for now.
# There is a lot going on here, but it's all necessary.
def perform_board_interventions(
    probe_names: dict[int, str],
    probe_data: train_test_chess.LinearProbeData,
    sample_size: int,
    intervention_type: InterventionType,
    sampling_type: SamplingType,
    recording_name: str,
    track_outputs: bool = False,
    scales: list[float] = [0.1],
):
    probes, state_stacks_all_chars, white_move_indices = prepare_intervention_data(
        probe_names, probe_data, sample_size
    )

    move_counters = MoveCounters()

    # Scale tracker stores the number of successful interventions for each scale
    deterministic_scale_tracker = {}
    sampled_scale_tracker = {}
    for scale in scales:
        deterministic_scale_tracker[scale] = 0
        sampled_scale_tracker[scale] = 0

    # Output tracker stores metadata and the original and modified probe outputs for the entire board per move per game for each layer
    # The results can be analyzed in probe_output_data_exploration.ipynb
    # CAUTION: This can quickly grow to gigabytes of data
    if track_outputs:
        output_tracker = initialize_output_tracker(probes)

    average_piece_values = {}
    average_blank_values = {}

    for sample_index in range(sample_size):
        for scale in scales:
            print(
                f"Scale: {scale}, deterministic count: {deterministic_scale_tracker[scale]}, sampled count: {sampled_scale_tracker[scale]}"
            )

        for move_of_interest in range(START_POS, END_POS):
            print(
                f"Sample index: {sample_index}, total moves: {move_counters.total_moves}, possible moves: {move_counters.possible_moves}, initial legal moves: {move_counters.original_board_deterministic_legal_moves}, legal intervention moves: {move_counters.modified_board_deterministic_legal_moves}"
            )
            move_counters.total_moves += 1

            # Step 1: Get the board state at move_of_interest
            move_of_interest_index = white_move_indices[sample_index][move_of_interest]
            pgn_string = probe_data.board_seqs_string[sample_index][
                : move_of_interest_index + 1
            ]
            board = chess_utils.pgn_string_to_board(pgn_string)

            # Step 2: Get the model move at move_of_interest
            # model_input.shape is (1, move_of_interest_index + 1)
            model_input = chess_utils.encode_string(META, pgn_string)
            model_input = torch.tensor(model_input).unsqueeze(0).to(device)
            deterministic_model_move = chess_utils.get_model_move(
                probe_data.model, META, model_input, temperature=0.0
            )

            # Step 3: Check if the model move is legal. parse_san will throw an exception if the move is illegal
            try:
                model_move_san = board.parse_san(deterministic_model_move)
            except:
                continue

            move_counters.original_board_deterministic_legal_moves += 1

            # Step 4: Determine which piece was moved from which source square
            moved_piece = board.piece_at(model_move_san.from_square)
            moved_piece_int = chess_utils.PIECE_TO_INT[moved_piece.piece_type]
            moved_piece_probe_index = chess_utils.PIECE_TO_ONE_HOT_MAPPING[
                moved_piece_int
            ]
            r, c = chess_utils.square_to_coordinate(model_move_san.from_square)

            # If the piece is a king, we skip the intervention as a legal chess game must have a king.
            if moved_piece.piece_type == chess.KING:
                continue

            # Step 5: Make a modified board where source square is now empty
            modified_board = board.copy()
            modified_board.set_piece_at(model_move_san.from_square, None)

            if not any(board.legal_moves):
                print("No legal moves available for the modified board. Skipping...")
                continue

            move_counters.possible_moves += 1

            # Step 5.1: Sample n moves from the unmodified model
            # Track how many moves were legal on both the original and modified boards
            for _ in range(SAMPLING_MOVES):
                if sampling_type == SamplingType.DETERMINISTIC_ONLY:
                    break
                sampled_model_move = chess_utils.get_model_move(
                    probe_data.model, META, model_input, temperature=TEMPERATURE
                )
                try:
                    board.parse_san(sampled_model_move)
                    print(f"Model original move: {sampled_model_move}")
                    move_counters.original_model_original_board_sampled_legal_moves += 1
                except:
                    print(f"Invalid original move: {sampled_model_move}")
                    pass
                try:
                    modified_board.parse_san(sampled_model_move)
                    print(f"Model modified move: {sampled_model_move}")
                    move_counters.original_model_modified_board_sampled_legal_moves += 1
                except:
                    print(f"Invalid modified move: {sampled_model_move}")
                    pass

            # If we are targetting probe output values, collect the average probe output values. Also, optionally track the outputs for each layer
            _, cache = probe_data.model.run_with_cache(model_input)

            # Initialize some legal move trackers. Note that these get reset every move. Add the end of the move,
            # we find the maximum value of these trackers to get the maximum possible legal moves for each move
            legal_deterministic_moves_per_scale_per_move = {}
            legal_sampled_moves_per_scale_per_move = {}
            for scale in scales:
                print(f"Scale: {scale}")
                legal_sampled_moves_per_scale_per_move[scale] = 0
                legal_deterministic_moves_per_scale_per_move[scale] = 0

                # Step 5.2: Calculate the average probe output values for non-blank and non-moved piece cells
                # Note that these are only used for the AVERAGE_TARGET intervention type
                # Also, optionally track the outputs for each layer
                if (
                    track_outputs
                    or intervention_type == InterventionType.AVERAGE_TARGET
                ):
                    probe_outputs = calculate_probe_outputs(probes, cache)
                    if intervention_type == InterventionType.AVERAGE_TARGET:
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

                    if track_outputs:
                        output_tracker = update_output_tracker_grids(
                            probe_outputs,
                            output_tracker,
                            layer,
                            move_of_interest_index,
                            r,
                            c,
                            moved_piece_probe_index,
                            deterministic_model_move,
                            "original",
                        )

                # This is the intervention function. In it, we obtain a vector to flip the square to blank in the model's activations at a given layer
                # Multiply it by some scale factor, then subtract it from the model's activations
                # If we make this function more modular and pass all variables in (probes, r, c, etc), it is much slower
                def flip_hook(
                    resid,
                    hook,
                    layer: int,
                    scale: float = 0.1,
                ):
                    target = 0.0
                    blank_probe = probes[layer][:, :, r, c, BLANK_INDEX].squeeze()
                    piece_probe = probes[layer][
                        :, :, r, c, moved_piece_probe_index
                    ].squeeze()

                    flip_dir = (piece_probe * 1.0) - (blank_probe * 1.0)

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
                            target,
                        )
                        scale = min(0.3, scale)

                    if layer < GPT_LAYER_COUNT:
                        # We want to intervene on multiple positions in the sequence because a move is multiple tokens
                        resid[:, :] -= scale * flip_dir

                    coeff = (
                        resid[0, move_of_interest_index] @ flip_dir / flip_dir.norm()
                    )

                    # So we only print once during inference
                    if resid.shape[1] <= move_of_interest_index + 1:
                        logger.debug(
                            f"Layer: {layer}, coeff: {coeff:10.3f}, scale: {scale:10.3f}, target: {target:10.3f}"
                        )

                # Step 6: Intervene on the model's activations and get the model move under the modified board state
                probe_data.model.reset_hooks()
                for layer in probes:
                    temp_hook_fn = partial(flip_hook, layer=layer, scale=scale)
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    probe_data.model.add_hook(hook_name, temp_hook_fn)

                modified_board_deterministic_model_move = chess_utils.get_model_move(
                    probe_data.model, META, model_input, temperature=0.0
                )

                # Step 6.1: Sample n moves from the modified model
                # Track how many moves were legal on the modified board
                # Note that we are tracking this for each scale
                for _ in range(SAMPLING_MOVES):
                    if sampling_type == SamplingType.DETERMINISTIC_ONLY:
                        break
                    sampled_model_move = chess_utils.get_model_move(
                        probe_data.model, META, model_input, temperature=TEMPERATURE
                    )
                    try:
                        modified_board.parse_san(sampled_model_move)
                        print(f"Model modified sampled move: {sampled_model_move}")
                        legal_sampled_moves_per_scale_per_move[scale] += 1
                    except:
                        print(f"Invalid modified sampled move: {sampled_model_move}")
                        pass

                _, modified_cache = probe_data.model.run_with_cache(model_input)
                probe_data.model.reset_hooks()

                # Step 6.2: If we are tracking outputs, update the output tracker with the modified outputs
                if track_outputs:
                    probe_outputs = calculate_probe_outputs(probes, modified_cache)
                    output_tracker = update_output_tracker_grids(
                        probe_outputs,
                        output_tracker,
                        layer,
                        move_of_interest_index,
                        r,
                        c,
                        moved_piece_probe_index,
                        modified_board_deterministic_model_move,
                        "modified",
                    )
                    for layer in output_tracker:
                        # Duplicating some metadata for each layer but it's a small amount
                        output_tracker[layer]["scales"].append(scale)
                        output_tracker[layer]["successes"].append(False)
                        output_tracker[layer]["cells"].append((r, c))
                        output_tracker[layer]["pieces"].append(moved_piece_int)
                        if intervention_type == InterventionType.AVERAGE_TARGET:
                            output_tracker[layer]["average_original_blank_grid"].append(
                                average_blank_values[layer]
                            )
                            output_tracker[layer]["average_original_piece_grid"].append(
                                average_piece_values[layer]
                            )

                # Step 7: Check if the model move under the modified board state is legal
                try:
                    modified_board.parse_san(modified_board_deterministic_model_move)
                    print(f"Modified move: {modified_board_deterministic_model_move}")
                except:
                    print(
                        f"Invalid modified move: {modified_board_deterministic_model_move}"
                    )
                    continue

                # Step 8: The move is legal. Update the legal move trackers
                if track_outputs:
                    for layer in output_tracker:
                        output_tracker[layer]["successes"][-1] = True
                legal_deterministic_moves_per_scale_per_move[scale] = 1
                deterministic_scale_tracker[scale] += 1
                sampled_scale_tracker[scale] += legal_sampled_moves_per_scale_per_move[
                    scale
                ]

            move_counters.modified_model_modified_board_sampled_legal_moves += max(
                legal_sampled_moves_per_scale_per_move.values()
            )
            move_counters.modified_board_deterministic_legal_moves += max(
                legal_deterministic_moves_per_scale_per_move.values()
            )

    # After intervening on all moves in all games, save output_tracker and move_counters to disk
    if track_outputs:
        file_path = "output_tracker.pkl"
        with open(file_path, "wb") as file:
            pickle.dump(output_tracker, file)
        print(f"File saved to {file_path}")
    for scale in scales:
        print(
            f"Scale: {scale}, deterministic count: {deterministic_scale_tracker[scale]}, sampled count: {sampled_scale_tracker[scale]}"
        )
    recording_name = RECORDING_DIR + "/" + recording_name + ".json"
    with open(recording_name, "w") as file:
        records = {}
        for field in fields(move_counters):
            records[field.name] = getattr(move_counters, field.name)
        records.update(deterministic_scale_tracker)
        records.update(sampled_scale_tracker)
        file.write(json.dumps(records))


scales_lookup = {
    InterventionType.SINGLE_SCALE: np.arange(0.05, 0.61, 0.1),
    InterventionType.AVERAGE_TARGET: np.arange(0.0, -12.1, -3.0),
    InterventionType.SINGLE_TARGET: np.arange(-2.0, -20.1, -4.0),
    # InterventionType.SINGLE_TARGET: np.arange(-0.1, 0.11, 0.05),
}

intervention_type = InterventionType.SINGLE_SCALE
intervention_type = InterventionType.AVERAGE_TARGET
# intervention_type = InterventionType.SINGLE_TARGET

intervention_types = [
    InterventionType.SINGLE_SCALE,
    InterventionType.AVERAGE_TARGET,
    InterventionType.SINGLE_TARGET,
]

sampling_type = SamplingType.BOTH
intervention_type = InterventionType.SINGLE_SCALE

probe_names = {}
first_layer = 3
last_layer = 8

# The last layer in the model has an average empty value about 2x of all other layers
# For simplicity, we will exclude the last layer from the intervention
if intervention_type == InterventionType.SINGLE_TARGET:
    last_layer = 7

for i in range(first_layer, last_layer):
    probe_names[i] = (
        f"tf_lens_lichess_8layers_ckpt_no_optimizer_chess_piece_probe_layer_{i}.pth"
    )
probe_data = get_probe_data(probe_names[first_layer])

scales = scales_lookup[intervention_type]

recording_name = f"intervention_type={intervention_type.value}_first_layer={first_layer}_last_layer={last_layer - 1}_scales="
for scale in scales:
    recording_name += f"{str(scale).replace('.', '')[:5]}_"

print(f"Recording name: {recording_name}")

perform_board_interventions(
    probe_names,
    probe_data,
    1,
    intervention_type,
    sampling_type,
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
#         sampling_type,
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
