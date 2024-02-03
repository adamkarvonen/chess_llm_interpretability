import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig
import einops
import torch
from tqdm import tqdm
import numpy as np
from fancy_einsum import einsum
import numpy as np
from dataclasses import dataclass
import pandas as pd
import logging
import itertools

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

batch_size = 1
MAXIMUM_TESTING_GAMES = 2000


def check_tensor_values(tensor, tensor_name="Tensor"):
    """Check if a tensor contains NaN, inf, or -inf values because we are summing 30k+ activations together."""
    # isneginf is currently not implemented for mps tensors
    original_device_type = tensor.device.type
    if original_device_type == "mps":
        tensor = tensor.cpu()

    if torch.any(torch.isinf(tensor)):
        raise ValueError(f"Overflow detected: {tensor_name} contains inf")
    if torch.any(torch.isneginf(tensor)):
        raise ValueError(f"Overflow detected: {tensor_name} contains -inf")
    if torch.any(torch.isnan(tensor)):
        raise ValueError(f"Invalid value detected: {tensor_name} contains NaN")

    if original_device_type == "mps":
        tensor = tensor.to("mps")


def create_contrastive_activations(
    activation_name: str,
    probe_data: LinearProbeData,
    config: Config,
    misc_logging_dict: dict,
):
    assert misc_logging_dict["split"] == "train", "Don't train on the test set"

    num_games = min(
        ((len(probe_data.board_seqs_int) // batch_size) * batch_size),
        (MAXIMUM_TESTING_GAMES // batch_size) * batch_size,
    )  # Unfortunately, num_games must be divisible by batch_size TODO: Fix this

    one_hot_range = config.max_val - config.min_val + 1
    if config.levels_of_interest is not None:
        one_hot_range = len(config.levels_of_interest)

    indexing_function_name = config.custom_indexing_function.__name__
    process_data = False
    if config.levels_of_interest is not None:
        process_data = True

    # logger.debug(dots_indices.shape)

    wandb_project = "chess_linear_probes"
    wandb_run_name = f"{config.linear_probe_name}_{model_name}_layer_{probe_data.layer}_indexing_{indexing_function_name}"
    if config.levels_of_interest is not None:
        wandb_run_name += "_levels"
        for level in config.levels_of_interest:
            wandb_run_name += f"_{level}"

    logging_dict = {
        "linear_probe_name": config.linear_probe_name,
        "model_name": model_name,
        "layer": probe_data.layer,
        "indexing_function_name": indexing_function_name,
        "batch_size": batch_size,
        "split": split,
        "pos_start": config.pos_start,
        "num_games": num_games,
        "one_hot_range": one_hot_range,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "config_name": config.linear_probe_name,
        "model_name": model_name,
        "dataset_prefix": dataset_prefix,
        "process_data": process_data,
        "column_name": config.column_name,
        "split": split,
        "levels_of_interest": config.levels_of_interest,
    }
    logging_dict.update(misc_logging_dict)

    current_iter = 0
    accuracy = 0
    accuracy_list = []
    with torch.inference_mode():
        full_train_indices = torch.arange(0, num_games)
        sum_high_elo = torch.zeros((512), device=device)
        sum_low_elo = torch.zeros((512), device=device)
        count_high_elo = 0
        count_low_elo = 0
        for i in tqdm(range(0, num_games, batch_size)):
            indices = full_train_indices[i : i + batch_size]
            list_of_indices = (
                indices.tolist()
            )  # For indexing into the board_seqs_string list of strings
            # logger.debug(list_of_indices)
            games_int = probe_data.board_seqs_int[indices]
            games_int = games_int[:, :]
            # logger.debug(games_int.shape)
            games_str = [probe_data.board_seqs_string[idx] for idx in list_of_indices]
            games_str = [s[:] for s in games_str]
            games_dots = probe_data.custom_indices[indices]
            games_dots = games_dots[:, config.pos_start :]
            # logger.debug(games_dots.shape)

            if config.probing_for_skill:
                games_skill = probe_data.skill_stack[indices]
                logger.debug(f"games_skill shape: {games_skill.shape}")
            else:
                raise Exception("CAA currently only supports skill vectors")

            # logger.debug(state_stack_one_hot.shape)
            _, cache = probe_data.model.run_with_cache(
                games_int.to(device)[:, :-1], return_type=None
            )
            resid_post = cache["resid_post", layer][:, :]
            # Initialize a list to hold the indexed state stacks
            indexed_resid_posts = []

            for batch_idx in range(games_dots.size(0)):
                # Get the indices for the current batch
                dots_indices_for_batch = games_dots[batch_idx]

                # Index the state_stack for the current batch
                indexed_resid_post = resid_post[batch_idx, dots_indices_for_batch]

                # Append the result to the list
                indexed_resid_posts.append(indexed_resid_post)

            # Stack the indexed state stacks along the first dimension
            # This results in a tensor of shape [2, 61, 8, 8] (assuming all batches have 61 indices)
            resid_post = torch.stack(indexed_resid_posts)
            summed_resid_post = einops.reduce(
                resid_post, "batch indices model_dim -> batch model_dim", "sum"
            )
            # logger.debug(f"resid_post: {resid_post.shape}")
            # logger.debug(f"summed_resid_post: {summed_resid_post.shape}")
            # logger.debug(resid_post[0, :, :2])
            # logger.debug(summed_resid_post[0, :2])

            for batch_idx in range(batch_size):
                if games_skill[batch_idx] == config.levels_of_interest[1]:
                    sum_high_elo += summed_resid_post[batch_idx]
                    count_high_elo += 1
                elif games_skill[batch_idx] == config.levels_of_interest[0]:
                    sum_low_elo += summed_resid_post[batch_idx]
                    count_low_elo += 1
                else:
                    raise Exception("Invalid skill level")

            logger.debug(
                f"count_high_elo: {count_high_elo}, count_low_elo: {count_low_elo}, games_skill: {games_skill}"
            )

            # assert resid_post.shape == state_stack_one_hot.shape

            if i % 100 == 0:
                logger.info(f"batch {i}, acc {accuracy}")

            current_iter += batch_size

            accuracy_list.append(accuracy)

    check_tensor_values(sum_high_elo, "sum_high_elo")
    check_tensor_values(sum_low_elo, "sum_low_elo")

    average_high_elo_activation = sum_high_elo / count_high_elo
    average_low_elo_activation = sum_low_elo / count_low_elo

    difference_vector = average_high_elo_activation - average_low_elo_activation

    state = {
        "average_high_elo_activation": average_high_elo_activation,
        "average_low_elo_activation": average_low_elo_activation,
        "difference_vector": difference_vector,
        "metadata": {
            "description": "Average vectors for high and low Elo chess games",
            "count_high_elo": count_high_elo,
            "count_low_elo": count_low_elo,
        },
    }
    state.update(logging_dict)

    output_probe_data_name = activation_name.split("/")[-1].split(".")[0]
    output_location = f"{CAA_DIR}{output_probe_data_name}.pt"

    logger.info(f"Saving activations to {output_location}")
    torch.save(state, output_location)


MODEL_DIR = "models/"
DATA_DIR = "data/"
CAA_DIR = "contrastive_activations/"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using device: {device}")

config = train_test_chess.skill_config

# Sweep over layers, levels of interest, pos_start, and dataset_prefix

layers = range(11, 16, 1)
levels_of_interest = [[1, 4]]
pos_starts = [25]
dataset_prefixes = [
    "lichess_",
]


for layer, level, pos_start, dataset_prefix in itertools.product(
    layers, levels_of_interest, pos_starts, dataset_prefixes
):
    dataset_prefix = dataset_prefix
    layer = layer
    split = "train"
    n_layers = 16
    model_name = f"tf_lens_{dataset_prefix}{n_layers}layers_ckpt_no_optimizer"
    config.levels_of_interest = level
    input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
    config = train_test_chess.set_config_min_max_vals_and_column_name(
        config, input_dataframe_file, dataset_prefix
    )
    config.pos_start = pos_start

    misc_logging_dict = {
        "split": split,
        "dataset_prefix": dataset_prefix,
        "model_name": model_name,
        "n_layers": n_layers,
    }

    probe_data = train_test_chess.construct_linear_probe_data(
        input_dataframe_file,
        layer,
        dataset_prefix,
        split,
        n_layers,
        model_name,
        config,
    )

    levels_str = [str(i) for i in level]
    levels_str = "".join(levels_str)

    activation_name = f"{dataset_prefix}{split}_layer_{layer}_pos_start_{config.pos_start}_levels_{levels_str}_activations_{MAXIMUM_TESTING_GAMES}_moves"
    create_contrastive_activations(
        activation_name, probe_data, config, misc_logging_dict
    )
