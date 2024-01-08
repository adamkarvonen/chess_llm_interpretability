# %%
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
from typing import Optional

import chess_utils

# %%
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

# %%
MODEL_DIR = "models/"
DATA_DIR = "data/"
PROBE_DIR = "linear_probes/"
SAVED_PROBE_DIR = "linear_probes/saved_probes/"
PROCESSING_DF_FILENAME = f"{DATA_DIR}temporary_in_processing.csv"
MAXIMUM_TRAINING_GAMES = 25000
MAXIMUM_TESTING_GAMES = 10000

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using device: {device}")


wandb_logging = False
os.environ["WANDB_MODE"] = "online"

# Training parameters
batch_size = 5
num_epochs = batch_size
modes = 1
wd = 0.01
max_lr = 3e-4
min_lr = max_lr / 10
decay_lr = True

# meta is used to encode the string pgn strings into integer sequences
with open(f"{MODEL_DIR}meta.pkl", "rb") as f:
    meta = pickle.load(f)

logger.info(meta)

stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

meta_round_trip_input = "1.e4 e6 2.Nf3"
logger.info(encode(meta_round_trip_input))
logger.info("Performing round trip test on meta")
assert decode(encode(meta_round_trip_input)) == meta_round_trip_input

# %%


# %%


@dataclass
class Config:
    min_val: int
    max_val: int
    custom_board_state_function: callable
    linear_probe_name: str
    custom_indexing_function: callable = chess_utils.find_dots_indices
    num_rows: int = 8
    num_cols: int = 8
    levels_of_interest: Optional[list[int]] = None
    column_name: str = None
    probing_for_skill: bool = False
    pos_start: int = 5  # indexes into custom_indexing_function. Example: if pos_start = 25, for find_dots_indices, selects everything after the first 25 moves


piece_config = Config(
    min_val=-6,
    max_val=6,
    custom_board_state_function=chess_utils.board_to_piece_state,
    linear_probe_name="chess_piece_probe",
)

color_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=chess_utils.board_to_piece_color_state,
    linear_probe_name="chess_color_probe",
)

random_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=chess_utils.board_to_random_state,
    linear_probe_name="chess_random_probe",
)

skill_config = Config(
    min_val=-2,
    max_val=20,
    custom_board_state_function=chess_utils.board_to_skill_state,
    linear_probe_name="chess_skill_probe",
    num_rows=1,
    num_cols=1,
    probing_for_skill=True,
    pos_start=25,
)


@dataclass
class LinearProbeData:
    layer: int
    model: HookedTransformer
    custom_indices: torch.tensor
    board_seqs_int: torch.tensor
    board_seqs_string: list[str]
    skill_stack: torch.tensor
    user_state_dict_one_hot_mapping: Optional[dict[int, int]] = None


def get_transformer_lens_model(model_name: str, n_layers: int) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=32,
        n_ctx=1023,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(f"{MODEL_DIR}{model_name}.pth"))
    model.to(device)
    return model


def process_dataframe(
    input_dataframe_file: str,
    config: Config,
) -> Optional[dict]:
    """This is used if we want to have our model do classification on a subset of the Elo bins.
    There are 6 Elo bins. If we want our model to classify between bin 0 and bin 5, we can use this function to
    filter the DataFrame to only include games from these bins."""
    df = pd.read_csv(input_dataframe_file)
    df.to_csv(PROCESSING_DF_FILENAME, index=False)
    user_state_dict_one_hot_mapping = None

    if config.levels_of_interest is not None:
        user_state_dict_one_hot_mapping = {}
        for i in range(len(config.levels_of_interest)):
            user_state_dict_one_hot_mapping[config.levels_of_interest[i]] = i

        matches = {number for number in config.levels_of_interest}
        logger.info(f"Levels to be used in probe dataset: {matches}")

        # Filter the DataFrame based on these matches
        filtered_df = df[df[config.column_name].isin(matches)]
        filtered_df.to_csv(PROCESSING_DF_FILENAME, index=False)
        logger.info(f"Number of games in filtered dataset: {len(filtered_df)}")

    return user_state_dict_one_hot_mapping


def get_board_seqs_string() -> list[str]:
    df = pd.read_csv(PROCESSING_DF_FILENAME)
    row_length = len(df["transcript"].iloc[0])
    num_games = len(df)

    assert all(
        df["transcript"].apply(lambda x: len(x) == row_length)
    ), "Not all transcripts are of length {}".format(row_length)

    board_seqs_string = df["transcript"]

    logger.info(
        f"Number of games: {len(board_seqs_string)},length of a game in chars: {len(board_seqs_string[0])}"
    )
    return board_seqs_string


def get_board_seqs_int() -> torch.tensor:
    df = pd.read_csv(PROCESSING_DF_FILENAME)
    encoded_df = df["transcript"].apply(encode)
    logger.info(encoded_df.head())
    board_seqs_int = torch.tensor(encoded_df.apply(list).tolist())
    logger.info(f"board_seqs_int shape: {board_seqs_int.shape}")
    return board_seqs_int


# %%


def get_skill_stack(config: Config) -> torch.tensor:
    df = pd.read_csv(f"{PROCESSING_DF_FILENAME}")
    skill_levels_list = df[config.column_name].tolist()

    skill_stack = torch.tensor(skill_levels_list)
    logger.info(f"Unique values in skill_stack: {skill_stack.unique()}")
    logger.info(f"skill_stack shape: {skill_stack.shape}")
    return skill_stack


def get_custom_indices(custom_indexing_function: callable) -> torch.tensor:
    custom_indices = chess_utils.find_custom_indices(
        PROCESSING_DF_FILENAME, custom_indexing_function
    )
    custom_indices = torch.tensor(custom_indices).long()
    logger.info(f"custom_indices shape: {custom_indices.shape}")
    return custom_indices


def get_lr(current_iter: int, max_iters: int, max_lr: float, min_lr: float) -> float:
    """
    Calculate the learning rate using linear decay.

    Args:
    - current_iter (int): The current iteration.
    - max_iters (int): The total number of iterations for decay.
    - lr (float): The initial learning rate.
    - min_lr (float): The minimum learning rate after decay.

    Returns:
    - float: The calculated learning rate.
    """
    # Ensure current_iter does not exceed max_iters
    current_iter = min(current_iter, max_iters)

    # Calculate the linearly decayed learning rate
    decayed_lr = max_lr - (max_lr - min_lr) * (current_iter / max_iters)

    return decayed_lr


# %%
def train_linear_probe_cross_entropy(
    probe_data: LinearProbeData,
    config: Config,
    misc_logging_dict: dict,
):
    """Trains a linear probe on the train set, contained in probe_data."""
    assert misc_logging_dict["split"] == "train", "Don't train on the test set"
    # I use min because the probes seem to mostly converge within 25k games
    num_games = min(
        ((len(probe_data.board_seqs_int) // batch_size) * batch_size),
        (MAXIMUM_TRAINING_GAMES // batch_size) * batch_size,
    )  # Unfortunately, num_games must be divisible by batch_size TODO: Fix this
    max_iters = num_games * num_epochs

    one_hot_range = config.max_val - config.min_val + 1
    if config.levels_of_interest is not None:
        one_hot_range = len(config.levels_of_interest)

    linear_probe_name = f"{PROBE_DIR}{model_name}_{config.linear_probe_name}_layer_{probe_data.layer}.pth"
    linear_probe = torch.randn(
        modes,
        probe_data.model.cfg.d_model,
        config.num_rows,
        config.num_cols,
        one_hot_range,
        requires_grad=False,
        device=device,
    ) / np.sqrt(probe_data.model.cfg.d_model)
    linear_probe.requires_grad = True
    logger.info(f"linear_probe shape: {linear_probe.shape}")
    lr = max_lr
    optimiser = torch.optim.AdamW(
        [linear_probe], lr=lr, betas=(0.9, 0.99), weight_decay=wd
    )

    logger.info(f"custom_indices shape: {probe_data.custom_indices.shape}")
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
        "max_lr": max_lr,
        "wd": wd,
        "split": split,
        "pos_start": config.pos_start,
        "num_epochs": num_epochs,
        "num_games": num_games,
        "modes": modes,
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

    if wandb_logging:
        import wandb

        wandb.init(project=wandb_project, name=wandb_run_name, config=logging_dict)

    current_iter = 0
    loss = 0
    accuracy = 0
    for epoch in range(num_epochs):
        full_train_indices = torch.randperm(num_games)
        for i in tqdm(range(0, num_games, batch_size)):
            lr = get_lr(current_iter, max_iters, max_lr, min_lr) if decay_lr else lr
            for param_group in optimiser.param_groups:
                param_group["lr"] = lr

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
                # logger.debug(games_skill.shape)
            else:
                games_skill = None

            state_stack = chess_utils.create_state_stacks(
                games_str, config.custom_board_state_function, games_skill
            )
            # logger.debug(state_stack.shape)
            indexed_state_stacks = []

            for batch_idx in range(batch_size):
                # Get the indices for the current batch
                dots_indices_for_batch = games_dots[batch_idx]

                # Index the state_stack for the current batch
                indexed_state_stack = state_stack[
                    :, batch_idx, dots_indices_for_batch, :, :
                ]

                # Append the result to the list
                indexed_state_stacks.append(indexed_state_stack)

            # Stack the indexed state stacks along the first dimension
            state_stack = torch.stack(indexed_state_stacks)

            # Use einops to rearrange the dimensions after stacking
            state_stack = einops.rearrange(
                state_stack, "batch modes pos row col -> modes batch pos row col"
            )

            # logger.debug("after indexing state stack shape", state_stack.shape)

            state_stack_one_hot = chess_utils.state_stack_to_one_hot(
                modes,
                config.num_rows,
                config.num_cols,
                config.min_val,
                config.max_val,
                device,
                state_stack,
                probe_data.user_state_dict_one_hot_mapping,
            ).to(device)

            # logger.debug(state_stack_one_hot.shape)
            with torch.inference_mode():
                _, cache = probe_data.model.run_with_cache(
                    games_int.to(device)[:, :-1], return_type=None
                )
                resid_post = cache["resid_post", probe_data.layer][:, :]
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
            resid_post = torch.stack(indexed_resid_posts)
            # logger.debug("Resid post", resid_post.shape)
            probe_out = einsum(
                "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
                resid_post,
                linear_probe,
            )
            logger.debug(
                f"probe_out: {probe_out.shape},state_stack_one_hot: {state_stack_one_hot.shape},state_stack: {state_stack.shape}"
            )

            assert probe_out.shape == state_stack_one_hot.shape

            accuracy = (
                (probe_out[0].argmax(-1) == state_stack_one_hot[0].argmax(-1))
                .float()
                .mean()
            )

            probe_log_probs = probe_out.log_softmax(-1)
            probe_correct_log_probs = (
                einops.reduce(
                    probe_log_probs * state_stack_one_hot,
                    "modes batch pos rows cols options -> modes pos rows cols",
                    "mean",
                )
                * one_hot_range
            )  # Multiply to correct for the mean over one_hot_range
            loss = -probe_correct_log_probs[0, :].mean(0).sum()

            loss.backward()
            if i % 100 == 0:
                logger.info(
                    f"epoch {epoch}, batch {i}, acc {accuracy}, loss {loss}, lr {lr}"
                )
                if wandb_logging:
                    wandb.log(
                        {
                            "acc": accuracy,
                            "loss": loss,
                            "lr": lr,
                            "epoch": epoch,
                            "iter": current_iter,
                        }
                    )

            optimiser.step()
            optimiser.zero_grad()
            current_iter += batch_size

    checkpoint = {
        "linear_probe": linear_probe,
        "final_loss": loss,
        "iters": current_iter,
        "epochs": epoch,
        "acc": accuracy,
    }
    # Update the checkpoint dictionary with the contents of logging_dict
    checkpoint.update(logging_dict)
    torch.save(checkpoint, linear_probe_name)


def construct_linear_probe_data(
    input_dataframe_file: str,
    layer: int,
    dataset_prefix: str,
    split: str,
    n_layers: int,
    model_name: str,
    config: Config,
) -> LinearProbeData:
    """We need the following data to train or test a linear probe:
    - The layer to probe in the GPT
    - The GPT model in transformer_lens format
    - The number of layers in the GPT
    - board_seqs_int: the integer sequences representing the chess games, encoded using meta.pkl
    - board_seqs_string: the string sequences representing the chess games
    - custom_indices: the indices of the moves we want to probe on. By default, these are the indices of every "."
    - skill_stack: the skill levels of the players in the games (only used if probing for skill)
    """
    # Checking for foot guns

    if dataset_prefix == "lichess_":
        assert (
            "stockfish" not in model_name
        ), "Are you sure you're using the right model?"

    if dataset_prefix == "stockfish_":
        assert "lichess" not in model_name, "Are you sure you're using the right model?"

    model = get_transformer_lens_model(model_name, n_layers)
    user_state_dict_one_hot_mapping = process_dataframe(input_dataframe_file, config)
    board_seqs_string = get_board_seqs_string()
    board_seqs_int = get_board_seqs_int()
    skill_stack = None
    if config.probing_for_skill:
        skill_stack = get_skill_stack(config)
    custom_indices = get_custom_indices(config.custom_indexing_function)

    game_length_in_chars = len(board_seqs_string[0])
    num_games = len(board_seqs_string)

    assert board_seqs_int.shape == (num_games, game_length_in_chars)

    if skill_stack is not None:
        assert skill_stack.shape == (num_games,)

    _, shortest_game_length_in_moves = custom_indices.shape
    assert custom_indices.shape == (num_games, shortest_game_length_in_moves)

    probe_data = LinearProbeData(
        layer=layer,
        model=model,
        custom_indices=custom_indices,
        board_seqs_int=board_seqs_int,
        board_seqs_string=board_seqs_string,
        skill_stack=skill_stack,
        user_state_dict_one_hot_mapping=user_state_dict_one_hot_mapping,
    )

    return probe_data


def set_config_min_max_vals_and_column_name(
    config: Config,
    input_dataframe_file: str,
    dataset_prefix: str,
) -> Config:
    if config.levels_of_interest is not None or config.probing_for_skill:
        if dataset_prefix == "stockfish_":
            config.column_name = "player_two"
        elif dataset_prefix == "lichess_":
            config.column_name = "WhiteEloBinIndex"
    else:
        return config
    df = pd.read_csv(input_dataframe_file)
    config.min_val = df[config.column_name].min()
    config.max_val = df[config.column_name].max()

    return config


# %%
def test_linear_probe_cross_entropy(
    linear_probe_name: str,
    probe_data: LinearProbeData,
    config: Config,
    misc_logging_dict: dict,
):
    """Takes a linear probe and tests it on the test set, contained in probe_data. Saves the results to a pickle file."""
    assert misc_logging_dict["split"] == "test", "Don't test on the train set"

    num_games = min(
        ((len(probe_data.board_seqs_int) // batch_size) * batch_size),
        (MAXIMUM_TESTING_GAMES // batch_size) * batch_size,
    )  # Unfortunately, num_games must be divisible by batch_size TODO: Fix this

    one_hot_range = config.max_val - config.min_val + 1
    if config.levels_of_interest is not None:
        one_hot_range = len(config.levels_of_interest)

    checkpoint = torch.load(linear_probe_name, map_location=device)
    linear_probe = checkpoint["linear_probe"]
    logger.info(f"linear_probe shape: {linear_probe.shape}")
    logger.info(f"custom_indices shape: {probe_data.custom_indices.shape}")
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
        "max_lr": max_lr,
        "wd": wd,
        "split": split,
        "pos_start": config.pos_start,
        "num_epochs": num_epochs,
        "num_games": num_games,
        "modes": modes,
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
    loss = 0
    accuracy = 0
    accuracy_list = []
    probe_out_list = (
        []
    )  # These are currently unused, they could be used to analyze error rate per turn or square
    state_stack_one_hot_list = []
    loss_list = []
    with torch.inference_mode():
        full_test_indices = torch.arange(0, num_games)
        for i in tqdm(range(0, num_games, batch_size)):
            indices = full_test_indices[i : i + batch_size]
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
                # logger.debug(games_skill.shape)
            else:
                games_skill = None

            state_stack = chess_utils.create_state_stacks(
                games_str, config.custom_board_state_function, games_skill
            )
            # state_stack = state_stack[:, pos_start:pos_end, :, :]
            # logger.debug(state_stack.shape)
            # Initialize a list to hold the indexed state stacks
            indexed_state_stacks = []

            for batch_idx in range(batch_size):
                # Get the indices for the current batch
                dots_indices_for_batch = games_dots[batch_idx]

                # Index the state_stack for the current batch
                indexed_state_stack = state_stack[
                    :, batch_idx, dots_indices_for_batch, :, :
                ]

                # Append the result to the list
                indexed_state_stacks.append(indexed_state_stack)

            # Stack the indexed state stacks along the first dimension
            # This results in a tensor of shape [2, 61, 8, 8] (assuming all batches have 61 indices)
            state_stack = torch.stack(indexed_state_stacks)

            # Use einops to rearrange the dimensions after stacking
            state_stack = einops.rearrange(
                state_stack, "batch modes pos row col -> modes batch pos row col"
            )

            # logger.debug("after indexing state stack shape", state_stack.shape)

            state_stack_one_hot = chess_utils.state_stack_to_one_hot(
                modes,
                config.num_rows,
                config.num_cols,
                config.min_val,
                config.max_val,
                device,
                state_stack,
                probe_data.user_state_dict_one_hot_mapping,
            ).to(device)

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
            # logger.debug("Resid post", resid_post.shape)
            probe_out = einsum(
                "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
                resid_post,
                linear_probe,
            )
            logger.debug(
                f"probe_out: {probe_out.shape},state_stack_one_hot: {state_stack_one_hot.shape},state_stack: {state_stack.shape}"
            )

            assert probe_out.shape == state_stack_one_hot.shape

            accuracy = (
                (probe_out[0].argmax(-1) == state_stack_one_hot[0].argmax(-1))
                .float()
                .mean()
            )

            probe_log_probs = probe_out.log_softmax(-1)
            probe_correct_log_probs = (
                einops.reduce(
                    probe_log_probs * state_stack_one_hot,
                    "modes batch pos rows cols options -> modes pos rows cols",
                    "mean",
                )
                * one_hot_range
            )  # Multiply to correct for the mean over one_hot_range
            loss = -probe_correct_log_probs[0, :].mean(0).sum()

            if i % 100 == 0:
                logger.info(f"batch {i}, acc {accuracy}, loss {loss}")

            current_iter += batch_size

            accuracy_list.append(accuracy)
            # probe_out_list.append(probe_out)
            # state_stack_one_hot_list.append(state_stack_one_hot)
            loss_list.append(loss)
    data = {
        "accuracy": accuracy_list,
        # "probe_out": probe_out_list,
        # "state_stack_one_hot": state_stack_one_hot_list,
        "loss": loss_list,
    }

    output_probe_data_name = linear_probe_name.split("/")[-1].split(".")[0]
    output_location = f"{PROBE_DIR}test_data/{output_probe_data_name}.pkl"

    logger.info(f"Saving test data to {output_location}")
    average_accuracy = sum(accuracy_list) / len(accuracy_list)
    logger.info(f"Average accuracy: {average_accuracy}")

    with open(output_location, "wb") as f:
        pickle.dump(data, f)


def find_config_by_name(config_name: str) -> Config:
    """
    Finds and returns the Config instance with a matching linear_probe_name.

    Args:
        config_name (str): The name of the config to search for.
        configs (List[Config]): The list of Config instances to search in.

    Returns:
        Optional[Config]: The matching Config instance, or None if not found.
    """
    all_configs = [piece_config, color_config, random_config, skill_config]
    for config in all_configs:
        if config.linear_probe_name == config_name:
            return config
    raise ValueError(f"Config with name {config_name} not found")


RUN_TEST_SET = True  # If True, we will test the probes on the test set. If False, we will train the probes on the train set
USE_PIECE_BOARD_STATE = True  # We will test or train a probe for piece board state
# If USE_PIECE_BOARD_STATE is False, we will test or train a probe on predicting player ELO

# If training a probe, make sure to set the below parameters in the else block

saved_piece_probe_name = "tf_lens_lichess_16layers_ckpt_no_optimizer_chess_piece_probe_layer_12_pos_start_0.pth"
saved_skill_probe_name = (
    "tf_lens_lichess_16layers_ckpt_no_optimizer_chess_skill_probe_layer_12.pth"
)

if __name__ == "__main__":
    if RUN_TEST_SET:
        # saved_probes = [
        #     file
        #     for file in os.listdir(SAVED_PROBE_DIR)
        #     if os.path.isfile(os.path.join(SAVED_PROBE_DIR, file))
        # ]
        saved_probes = []

        # Quick and janky way to select between piece and skill probes
        if USE_PIECE_BOARD_STATE:
            saved_probes.append(saved_piece_probe_name)
        else:
            saved_probes.append(saved_skill_probe_name)

        print(saved_probes)

        for probe_to_test in saved_probes:
            probe_file_location = f"{SAVED_PROBE_DIR}{probe_to_test}"
            # We will populate all parameters using information in the probe state dict
            with open(probe_file_location, "rb") as f:
                state_dict = torch.load(f, map_location=torch.device(device))
                print(state_dict.keys())
                for key in state_dict.keys():
                    if key != "linear_probe":
                        print(key, state_dict[key])

                config = find_config_by_name(state_dict["config_name"])
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

                split = "test"
                input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
                config = set_config_min_max_vals_and_column_name(
                    config, input_dataframe_file, dataset_prefix
                )
                misc_logging_dict = {
                    "split": split,
                    "dataset_prefix": dataset_prefix,
                    "model_name": model_name,
                    "n_layers": n_layers,
                }

                probe_data = construct_linear_probe_data(
                    input_dataframe_file,
                    layer,
                    dataset_prefix,
                    split,
                    n_layers,
                    model_name,
                    config,
                )
                test_linear_probe_cross_entropy(
                    probe_file_location, probe_data, config, misc_logging_dict
                )
    else:
        # Quick and janky way to select between piece and skill configs
        config = None
        if USE_PIECE_BOARD_STATE:
            config = piece_config
        else:
            config = skill_config

        # When training a probe, you have to set all parameters such as model name, dataset prefix, etc.
        dataset_prefix = "lichess_"
        # dataset_prefix = "stockfish_"
        layer = 12
        split = "train"
        n_layers = 16
        model_name = f"tf_lens_{dataset_prefix}{n_layers}layers_ckpt_no_optimizer"
        # model_name = "tf_lens_lichess_16layers_ckpt_no_optimizer"
        # config.levels_of_interest = [0, 5] # NOTE: If training for skill, you should uncomment this line for good results
        input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
        config = set_config_min_max_vals_and_column_name(
            config, input_dataframe_file, dataset_prefix
        )

        misc_logging_dict = {
            "split": split,
            "dataset_prefix": dataset_prefix,
            "model_name": model_name,
            "n_layers": n_layers,
        }

        probe_data = construct_linear_probe_data(
            input_dataframe_file,
            layer,
            dataset_prefix,
            split,
            n_layers,
            model_name,
            config,
        )
        train_linear_probe_cross_entropy(probe_data, config, misc_logging_dict)
