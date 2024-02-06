import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig
import einops
import torch
from tqdm import tqdm
import numpy as np
from fancy_einsum import einsum
import chess
import numpy as np
import csv
from dataclasses import dataclass
from torch.nn import MSELoss, L1Loss
import pandas as pd
import pickle
import os
import logging
import plotly.graph_objects as go

import chess_utils
import train_test_chess
from train_test_chess import Config, LinearProbeData

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
SPLIT = "test"
MODES = 1  # Currently only supporting 1 mode so this is fairly unnecessary
START_POS = 5
END_POS = 30
BLANK_INDEX = chess_utils.PIECE_TO_ONE_HOT_MAPPING[0]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
logger.info(f"Using device: {device}")

with open("models/meta.pkl", "rb") as f:
    META = pickle.load(f)


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


def perform_board_interventions(
    probe_name: str,
    probe_data: train_test_chess.LinearProbeData,
    sample_size: int = 100,
):
    probe_file_location = f"{SAVED_PROBE_DIR}{probe_name}"
    checkpoint = torch.load(probe_file_location, map_location=torch.device(device))
    linear_probe = checkpoint["linear_probe"]

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

    indexed_state_stacks = []

    for batch_idx in range(sample_size):
        # Get the indices for the current batch
        dots_indices_for_batch = white_move_indices[batch_idx]

        # Index the state_stack for the current batch. Adding an unsqueeze operation to maintain the batch dimension.
        indexed_state_stack = state_stacks_all_chars[
            :, batch_idx : batch_idx + 1, dots_indices_for_batch, :, :
        ]

        # Append the result to the list
        indexed_state_stacks.append(indexed_state_stack)

    # Concatenate the indexed state stacks along the second dimension (batch dimension)
    # Since we're maintaining the batch dimension during indexing, we don't need to add it back in.
    state_stack_white_moves = torch.cat(indexed_state_stacks, dim=1)

    print("state stack shapes")
    print(state_stack_white_moves.shape)
    print(state_stacks_all_chars.shape)

    one_hot_range = config.max_val - config.min_val + 1
    board_seqs_int = probe_data.board_seqs_int[:sample_size].to(device)

    state_stacks_one_hot = chess_utils.state_stack_to_one_hot(
        MODES,
        config.num_rows,
        config.num_cols,
        config.min_val,
        config.max_val,
        device,
        state_stacks_all_chars,
    )

    hook_name = f"blocks.{probe_data.layer}.hook_resid_post"

    total_moves_counter = 0
    possible_moves_counter = 0
    initial_legal_moves_counter = 0
    legal_intervention_moves_counter = 0

    for sample_index in range(sample_size):
        print(
            f"Sample index: {sample_index}, total moves: {total_moves_counter}, possible moves: {possible_moves_counter}, initial legal moves: {initial_legal_moves_counter}, legal intervention moves: {legal_intervention_moves_counter}"
        )
        for move_of_interest in range(START_POS, END_POS):
            print(
                f"Sample index: {sample_index}, total moves: {total_moves_counter}, possible moves: {possible_moves_counter}, initial legal moves: {initial_legal_moves_counter}, legal intervention moves: {legal_intervention_moves_counter}"
            )
            total_moves_counter += 1

            # Step 1: Get the board state at move_of_interest
            move_of_interest_index = white_move_indices[sample_index][move_of_interest]
            move_of_interest_state = state_stacks_all_chars[0][sample_index][
                move_of_interest_index
            ]
            board = chess_utils.state_stack_to_chess_board(move_of_interest_state)

            # Step 2: Get the model move at move_of_interest
            model_input = board_seqs_int[
                sample_index, : move_of_interest_index + 1
            ].unsqueeze(0)
            # model_input.shape is (1, move_of_interest_index + 1)
            model_move = chess_utils.get_model_move(probe_data.model, META, model_input)

            try:
                model_move_san = board.parse_san(model_move)
            except:
                continue

            # Step 3: Check if the model move is legal
            if model_move_san in board.legal_moves:
                initial_legal_moves_counter += 1
            else:
                continue

            # Step 4: Determine which piece was moved from which source square
            moved_piece = board.piece_at(model_move_san.from_square)
            moved_piece_int = chess_utils.PIECE_TO_INT[moved_piece.piece_type]
            moved_piece_probe_index = chess_utils.PIECE_TO_ONE_HOT_MAPPING[
                moved_piece_int
            ]
            r, c = chess_utils.square_to_coordinate(model_move_san.from_square)

            if moved_piece.piece_type == chess.KING:
                continue
            possible_moves_counter += 1

            # Step 5: Make a modified state_stack and board where source square is now empty
            modified_state_stack = state_stacks_all_chars[
                :, sample_index : sample_index + 1, :, :, :
            ]
            modified_state_stack[0, 0, move_of_interest_index, r, c] = 0
            modified_move_of_interest_state = modified_state_stack[
                0, 0, move_of_interest_index
            ]
            modified_board = chess_utils.state_stack_to_chess_board(
                modified_move_of_interest_state
            )

            if any(move for move in board.legal_moves) == False:
                print("No legal moves")
                total_moves_counter -= 1
                initial_legal_moves_counter -= 1
                continue

            # Step 6: Obtain a vector to flip the square to blank in the model's activations at a given layer
            piece1 = BLANK_INDEX
            piece2 = moved_piece_probe_index
            piece1_probe = linear_probe[:, :, r, c, piece1].squeeze()
            piece2_probe = linear_probe[:, :, r, c, piece2].squeeze()
            flip_dir = piece2_probe - piece1_probe
            flip_dir.to(device)

            def flip_hook(resid, hook):
                coeff = resid[0, move_of_interest_index] @ flip_dir / flip_dir.norm()
                # if coeff.item() > 0:
                # coeff = 1
                # print(f"coeff: {coeff}")
                coeff = min(5, (1 / abs(coeff.item())))
                resid[0,] -= (1.0) * coeff * flip_dir / flip_dir.norm()
                # resid[0, :] -= (
                #     flip_dir * 0.2
                # )  # NOTE: We could only intervene on a single position in the sequence, but there's no harm in intervening on all of them

            # Step 7: Intervene on the model's activations and get the model move under the modified board state
            probe_data.model.reset_hooks()
            probe_data.model.add_hook(hook_name, flip_hook)
            modified_board_model_move = chess_utils.get_model_move(
                probe_data.model, META, model_input
            )
            probe_data.model.reset_hooks()

            try:
                modified_board_model_move_san = modified_board.parse_san(
                    modified_board_model_move
                )
            except:
                print(board)
                print("_" * 50)
                print(modified_board)
                print(
                    f"Model move: {model_move}, Modified model move: {modified_board_model_move}"
                )
                print(
                    f"Model move: {model_move_san}, Modified model move: {modified_board_model_move_san}"
                )
                print(
                    f"Move of interest index: {move_of_interest_index}, r: {r}, c: {c}, Piece: {moved_piece}"
                )
                continue

            # Step 8: Check if the model move under the modified board state is legal
            if modified_board_model_move_san in modified_board.legal_moves:
                legal_intervention_moves_counter += 1


probe_name = "tf_lens_lichess_8layers_ckpt_no_optimizer_chess_piece_probe_layer_4.pth"
probe_data = get_probe_data(probe_name)
perform_board_interventions(probe_name, probe_data)
