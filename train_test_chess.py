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
from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
from beartype import beartype

import chess_utils

# %%
# Flags to control logging
debug_mode = False
info_mode = True

logger = logging.getLogger(__name__)

if debug_mode:
    logger.setLevel(logging.DEBUG)
elif info_mode:
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.WARNING)

# Add handler to this logger if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# %%
MODEL_DIR = "models/"
DATA_DIR = "data/"
PROBE_DIR = "linear_probes/"
SAVED_PROBE_DIR = "linear_probes/saved_probes/"
WANDB_PROJECT = "chess_linear_probes"
BATCH_SIZE = 2

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
logger.info(f"Using device: {DEVICE}")


wandb_logging = False
os.environ["WANDB_MODE"] = "online"

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
class TrainingParams:
    modes: int = 1
    # modes currently doesn't do anything, but it is used and adds a dimension to the tensors
    # In the future, modes could be used to do clever things like training multiple probes at once, such as a black piece probe and a white piece probe
    wd: float = 0.01
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.99
    max_train_games: int = 10000
    max_test_games: int = 10000
    max_val_games: int = 1000
    max_iters: int = 50000
    eval_iters: int = 50
    num_epochs: int = max_iters // max_train_games


TRAIN_PARAMS = TrainingParams()


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
    # pos_start indexes into custom_indexing_function. Example: if pos_start = 25, for find_dots_indices, selects everything after the first 25 moves
    pos_start: int = 0
    # If pos_end is None, it's set to the length of the shortest game in construct_linear_probe_data()
    pos_end: Optional[int] = None


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


def get_transformer_lens_model(
    model_name: str, n_layers: int, device: torch.device
) -> HookedTransformer:
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
) -> tuple[Optional[dict], pd.DataFrame]:
    """This is used if we want to have our model do classification on a subset of the Elo bins.
    There are 6 Elo bins. If we want our model to classify between bin 0 and bin 5, we can use this function to
    filter the DataFrame to only include games from these bins."""
    df = pd.read_csv(input_dataframe_file)
    user_state_dict_one_hot_mapping = None

    if config.levels_of_interest is not None:
        user_state_dict_one_hot_mapping = {}
        for i in range(len(config.levels_of_interest)):
            user_state_dict_one_hot_mapping[config.levels_of_interest[i]] = i

        matches = {number for number in config.levels_of_interest}
        logger.info(f"Levels to be used in probe dataset: {matches}")

        # Filter the DataFrame based on these matches
        df = df[df[config.column_name].isin(matches)]
        logger.info(f"Number of games in filtered dataset: {len(df)}")

        df = df.reset_index(drop=True)

    return user_state_dict_one_hot_mapping, df


def init_logging_dict(
    config: Config,
    probe_data: LinearProbeData,
    split: str,
    dataset_prefix: str,
    model_name: str,
    n_layers: int,
) -> dict:

    indexing_function_name = config.custom_indexing_function.__name__

    wandb_run_name = f"{config.linear_probe_name}_{model_name}_layer_{probe_data.layer}_indexing_{indexing_function_name}_max_games_{TRAIN_PARAMS.max_train_games}"
    if config.levels_of_interest is not None:
        wandb_run_name += "_levels"
        for level in config.levels_of_interest:
            wandb_run_name += f"_{level}"

    logging_dict = {
        "linear_probe_name": config.linear_probe_name,
        "layer": probe_data.layer,
        "indexing_function_name": indexing_function_name,
        "batch_size": BATCH_SIZE,
        "lr": TRAIN_PARAMS.lr,
        "wd": TRAIN_PARAMS.wd,
        "pos_start": config.pos_start,
        "num_epochs": TRAIN_PARAMS.num_epochs,
        "modes": TRAIN_PARAMS.modes,
        "wandb_project": WANDB_PROJECT,
        "config_name": config.linear_probe_name,
        "column_name": config.column_name,
        "levels_of_interest": config.levels_of_interest,
        "split": split,
        "dataset_prefix": dataset_prefix,
        "model_name": model_name,
        "n_layers": n_layers,
        "wandb_run_name": wandb_run_name,
    }

    return logging_dict


def get_board_seqs_string(df: pd.DataFrame) -> list[str]:
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


@jaxtyped(typechecker=beartype)
def get_board_seqs_int(df: pd.DataFrame) -> Int[Tensor, "num_games pgn_str_length"]:
    encoded_df = df["transcript"].apply(encode)
    logger.info(encoded_df.head())
    board_seqs_int = torch.tensor(encoded_df.apply(list).tolist())
    logger.info(f"board_seqs_int shape: {board_seqs_int.shape}")
    return board_seqs_int


# %%


@jaxtyped(typechecker=beartype)
def get_skill_stack(config: Config, df: pd.DataFrame) -> Int[Tensor, "num_games"]:
    skill_levels_list = df[config.column_name].tolist()

    skill_stack = torch.tensor(skill_levels_list)
    logger.info(f"Unique values in skill_stack: {skill_stack.unique()}")
    logger.info(f"skill_stack shape: {skill_stack.shape}")
    return skill_stack


# @jaxtyped(typechecker=beartype) # typechecking not supported for callable
def get_custom_indices(
    custom_indexing_function: callable, df: pd.DataFrame
) -> Int[Tensor, "num_games num_white_moves"]:
    custom_indices = chess_utils.find_custom_indices(custom_indexing_function, df)
    custom_indices = torch.tensor(custom_indices).long()
    logger.info(f"custom_indices shape: {custom_indices.shape}")
    return custom_indices


@jaxtyped(typechecker=beartype)
def prepare_data_batch(
    indices: Int[Tensor, "batch_size"], probe_data: LinearProbeData, config: Config
) -> tuple[
    Int[Tensor, "modes batch_size num_white_moves num_rows num_cols num_options"],
    Float[Tensor, "batch_size num_white_moves d_model"],
]:
    list_of_indices = indices.tolist()  # For indexing into the board_seqs_string list of strings
    games_int = probe_data.board_seqs_int[indices]  # games_int shape (batch_size, pgn_str_length)
    games_str = [probe_data.board_seqs_string[idx] for idx in list_of_indices]
    games_str = [s[:] for s in games_str]
    games_dots = probe_data.custom_indices[indices]
    games_dots = games_dots[
        :, config.pos_start : config.pos_end
    ]  # games_dots shape (batch_size, num_white_moves)

    if config.probing_for_skill:
        games_skill = probe_data.skill_stack[indices]  # games_skill shape (batch_size,)
    else:
        games_skill = None

    state_stack = chess_utils.create_state_stacks(
        games_str, config.custom_board_state_function, games_skill
    )  # shape (modes, batch_size, pgn_str_length, num_rows, num_cols)
    indexed_state_stacks = []

    for batch_idx in range(BATCH_SIZE):
        # Get the indices for the current batch
        dots_indices_for_batch = games_dots[batch_idx]

        # Index the state_stack for the current batch
        indexed_state_stack = state_stack[:, batch_idx, dots_indices_for_batch, :, :]

        # Append the result to the list
        indexed_state_stacks.append(indexed_state_stack)

    # Stack the indexed state stacks along the first dimension
    state_stack = torch.stack(
        indexed_state_stacks
    )  # shape (batch_size, modes, num_white_moves, num_rows, num_cols)

    # Use einops to rearrange the dimensions after stacking
    state_stack = einops.rearrange(
        state_stack, "batch modes pos row col -> modes batch pos row col"
    )  # shape (modes, batch_size, num_white_moves, num_rows, num_cols)

    state_stack_one_hot = chess_utils.state_stack_to_one_hot(
        TRAIN_PARAMS.modes,
        config.num_rows,
        config.num_cols,
        config.min_val,
        config.max_val,
        DEVICE,
        state_stack,
        probe_data.user_state_dict_one_hot_mapping,
    ).to(
        DEVICE
    )  # shape (modes, batch_size, num_white_moves, num_rows, num_cols, num_options)

    with torch.inference_mode():
        _, cache = probe_data.model.run_with_cache(games_int.to(DEVICE)[:, :-1], return_type=None)
        resid_post = cache["resid_post", probe_data.layer][
            :, :
        ]  # shape (batch_size, pgn_str_length - 1, d_model)
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
    resid_post = torch.stack(indexed_resid_posts)  # shape (batch_size, num_white_moves, d_model)

    return state_stack_one_hot, resid_post


@jaxtyped(typechecker=beartype)
def linear_probe_forward_pass(
    linear_probe: Float[Tensor, "modes d_model rows cols options"],
    state_stack_one_hot: Int[Tensor, "modes batch num_white_moves rows cols options"],
    resid_post: Float[Tensor, "batch num_white_moves d_model"],
    one_hot_range: int,
) -> tuple[Tensor, Tensor]:
    """Outputs are scalar tensors."""
    probe_out = einsum(
        "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
        resid_post,
        linear_probe,
    )

    assert probe_out.shape == state_stack_one_hot.shape

    accuracy = (probe_out[0].argmax(-1) == state_stack_one_hot[0].argmax(-1)).float().mean()

    probe_log_probs = probe_out.log_softmax(-1)
    probe_correct_log_probs = (
        einops.reduce(
            probe_log_probs * state_stack_one_hot,
            "modes batch pos rows cols options -> modes pos rows cols",
            "mean",
        )
        * one_hot_range
    )  # Multiply to correct for the mean over one_hot_range
    # probe_correct_log_probs shape (modes, num_white_moves, num_rows, num_cols)
    loss = -probe_correct_log_probs[0, :].mean(0).sum()

    return loss, accuracy


# helps estimate an arbitrarily accurate loss over either split using many batches
# This is mainly useful for checking that the probe isn't overfitting to the train set
# Note that I'm not doing a proper train/val split here, this was just a quick and dirty way to check for overfitting
@torch.no_grad()
def estimate_loss(
    train_games: int,
    val_games: int,
    linear_probe: Float[Tensor, "modes d_model rows cols options"],
    probe_data: LinearProbeData,
    config: Config,
    one_hot_range: int,
) -> dict[str, dict[str, float]]:
    out = {"train": {"loss": 0, "accuracy": 0}, "val": {"loss": 0, "accuracy": 0}}

    eval_iters = (TRAIN_PARAMS.eval_iters // BATCH_SIZE) * BATCH_SIZE

    train_indices = torch.randperm(train_games)[:eval_iters]
    val_indices = torch.randperm(val_games) + train_games  # to avoid overlap
    val_indices = val_indices[:eval_iters]

    split_indices = {"train": train_indices, "val": val_indices}
    for split in split_indices:
        losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        for k in range(0, eval_iters, BATCH_SIZE):
            indices = split_indices[split][k : k + BATCH_SIZE]

            state_stack_one_hot, resid_post = prepare_data_batch(indices, probe_data, config)

            loss, accuracy = linear_probe_forward_pass(
                linear_probe, state_stack_one_hot, resid_post, one_hot_range
            )
            losses[k : k + BATCH_SIZE] = loss.item()
            accuracies[k : k + BATCH_SIZE] = accuracy.item()
        out[split]["loss"] = losses.mean()
        out[split]["accuracy"] = accuracies.mean()
    return out


# %%
def train_linear_probe_cross_entropy(
    probe_data: LinearProbeData,
    config: Config,
    logging_dict: dict,
):
    """Trains a linear probe on the train set, contained in probe_data."""
    assert logging_dict["split"] == "train", "Don't train on the test set"

    val_games = (TRAIN_PARAMS.max_val_games // BATCH_SIZE) * BATCH_SIZE
    train_games = (TRAIN_PARAMS.max_train_games // BATCH_SIZE) * BATCH_SIZE

    num_games = train_games + val_games

    if len(probe_data.board_seqs_int) < num_games:
        raise ValueError("Not enough games to train on")
        # We raise an error so it doesn't fail silently. If we want to use less games, we can comment the error out
        # and add some logic to set train and val games to the number of games we have

    logging_dict["num_games"] = num_games

    one_hot_range = config.max_val - config.min_val + 1
    if config.levels_of_interest is not None:
        one_hot_range = len(config.levels_of_interest)

    linear_probe_name = (
        f"{PROBE_DIR}{model_name}_{config.linear_probe_name}_layer_{probe_data.layer}.pth"
    )
    linear_probe = torch.randn(
        TRAIN_PARAMS.modes,
        probe_data.model.cfg.d_model,
        config.num_rows,
        config.num_cols,
        one_hot_range,
        requires_grad=False,
        device=DEVICE,
    ) / np.sqrt(probe_data.model.cfg.d_model)
    linear_probe.requires_grad = True
    logger.info(f"linear_probe shape: {linear_probe.shape}")

    optimiser = torch.optim.AdamW(
        [linear_probe],
        lr=TRAIN_PARAMS.lr,
        betas=(TRAIN_PARAMS.beta1, TRAIN_PARAMS.beta2),
        weight_decay=TRAIN_PARAMS.wd,
    )

    logger.info(f"custom_indices shape: {probe_data.custom_indices.shape}")

    if wandb_logging:
        import wandb

        wandb.init(
            project=WANDB_PROJECT,
            name=logging_dict["wandb_run_name"],
            config=logging_dict,
        )

    current_iter = 0
    loss = 0
    accuracy = 0
    for epoch in range(TRAIN_PARAMS.num_epochs):
        full_train_indices = torch.randperm(train_games)
        for i in tqdm(range(0, train_games, BATCH_SIZE)):

            indices = full_train_indices[i : i + BATCH_SIZE]  # shape batch_size

            state_stack_one_hot, resid_post = prepare_data_batch(indices, probe_data, config)

            loss, accuracy = linear_probe_forward_pass(
                linear_probe, state_stack_one_hot, resid_post, one_hot_range
            )

            loss.backward()
            if i % 100 == 0:
                logger.info(f"epoch {epoch}, batch {i}, acc {accuracy}, loss {loss}")
                if wandb_logging:
                    wandb.log(
                        {
                            "acc": accuracy,
                            "loss": loss,
                            "epoch": epoch,
                            "iter": current_iter,
                        }
                    )

            optimiser.step()
            optimiser.zero_grad()
            current_iter += BATCH_SIZE

            if current_iter % 1000 == 0:
                losses = estimate_loss(
                    train_games,
                    val_games,
                    linear_probe,
                    probe_data,
                    config,
                    one_hot_range,
                )
                logger.info(f"epoch {epoch}, losses: {losses}, accuracy: {accuracy}")
                if wandb_logging:
                    wandb.log(
                        {
                            "train_loss": losses["train"]["loss"],
                            "train_acc": losses["train"]["accuracy"],
                            "val_loss": losses["val"]["loss"],
                            "val_acc": losses["val"]["accuracy"],
                            "epoch": epoch,
                        }
                    )
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
    n_layers: int,
    model_name: str,
    config: Config,
    max_games: int,
    device: torch.device,
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
        assert "stockfish" not in model_name, "Are you sure you're using the right model?"

    if dataset_prefix == "stockfish_":
        assert "lichess" not in model_name, "Are you sure you're using the right model?"

    model = get_transformer_lens_model(model_name, n_layers, device)
    user_state_dict_one_hot_mapping, df = process_dataframe(input_dataframe_file, config)
    df = df[:max_games]
    board_seqs_string = get_board_seqs_string(df)
    board_seqs_int = get_board_seqs_int(df)
    skill_stack = None
    if config.probing_for_skill:
        skill_stack = get_skill_stack(config, df)
    custom_indices = get_custom_indices(config.custom_indexing_function, df)

    pgn_str_length = len(board_seqs_string[0])
    num_games = len(board_seqs_string)

    assert board_seqs_int.shape == (num_games, pgn_str_length)

    if skill_stack is not None:
        assert skill_stack.shape == (num_games,)

    _, shortest_game_length_in_moves = custom_indices.shape
    assert custom_indices.shape == (num_games, shortest_game_length_in_moves)

    if not config.pos_end:
        config.pos_end = shortest_game_length_in_moves

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
        elif "lichess_" in dataset_prefix:
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
    logging_dict: dict,
):
    """Takes a linear probe and tests it on the test set, contained in probe_data. Saves the results to a pickle file."""
    assert logging_dict["split"] == "test", "Don't test on the train set"

    num_games = (TRAIN_PARAMS.max_test_games // BATCH_SIZE) * BATCH_SIZE

    if (len(probe_data.board_seqs_int) // BATCH_SIZE) * BATCH_SIZE < num_games:
        raise ValueError("Not enough games to test on")
        # We raise an error so it doesn't fail silently. If we want to use less games, we can comment the error out
        num_games = (len(probe_data.board_seqs_int) // BATCH_SIZE) * BATCH_SIZE

    one_hot_range = config.max_val - config.min_val + 1
    if config.levels_of_interest is not None:
        one_hot_range = len(config.levels_of_interest)

    logging_dict["num_games"] = num_games

    checkpoint = torch.load(linear_probe_name, map_location=DEVICE)
    linear_probe = checkpoint["linear_probe"]
    logger.info(f"linear_probe shape: {linear_probe.shape}")
    logger.info(f"custom_indices shape: {probe_data.custom_indices.shape}")

    current_iter = 0
    loss = 0
    accuracy = 0
    accuracy_list = []
    loss_list = []
    with torch.inference_mode():
        full_test_indices = torch.arange(0, num_games)
        for i in tqdm(range(0, num_games, BATCH_SIZE)):
            indices = full_test_indices[i : i + BATCH_SIZE]  # shape batch_size

            state_stack_one_hot, resid_post = prepare_data_batch(indices, probe_data, config)

            loss, accuracy = linear_probe_forward_pass(
                linear_probe, state_stack_one_hot, resid_post, one_hot_range
            )

            accuracy_list.append(accuracy.item())
            loss_list.append(loss.item())

            if i % 100 == 0:
                average_accuracy = sum(accuracy_list) / len(accuracy_list)
                logger.info(
                    f"batch {i}, average accuracy: {average_accuracy}, acc {accuracy}, loss {loss}"
                )

            current_iter += BATCH_SIZE
    data = {
        "accuracy": accuracy_list,
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


RUN_TEST_SET = False  # If True, we will test the probes on the test set. If False, we will train the probes on the train set
USE_PIECE_BOARD_STATE = True  # We will test or train a probe for piece board state
# If USE_PIECE_BOARD_STATE is False, we will test or train a probe on predicting player ELO

# If training a probe, make sure to set the below parameters in the else block

saved_piece_probe_name = (
    "tf_lens_lichess_16layers_ckpt_no_optimizer_chess_piece_probe_layer_12_pos_start_0.pth"
)
saved_skill_probe_name = "tf_lens_lichess_16layers_ckpt_no_optimizer_chess_skill_probe_layer_12.pth"

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
                state_dict = torch.load(f, map_location=torch.device(DEVICE))
                print(state_dict.keys())
                for key in state_dict.keys():
                    if key != "linear_probe":
                        print(key, state_dict[key])

                config = find_config_by_name(state_dict["config_name"])
                layer = state_dict["layer"]
                model_name = state_dict["model_name"]
                dataset_prefix = state_dict["dataset_prefix"]
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

                probe_data = construct_linear_probe_data(
                    input_dataframe_file,
                    layer,
                    dataset_prefix,
                    n_layers,
                    model_name,
                    config,
                    TRAIN_PARAMS.max_test_games,
                    DEVICE,
                )

                logging_dict = init_logging_dict(
                    config, probe_data, split, dataset_prefix, model_name, n_layers
                )

                test_linear_probe_cross_entropy(
                    probe_file_location, probe_data, config, logging_dict
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
        layer = 5
        split = "train"
        n_layers = 8
        model_name = f"tf_lens_{dataset_prefix}{n_layers}layers_ckpt_no_optimizer"

        # If probing for skill, set the levels of interest by default
        if not USE_PIECE_BOARD_STATE:
            config.levels_of_interest = [0, 5]

        input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
        config = set_config_min_max_vals_and_column_name(
            config, input_dataframe_file, dataset_prefix
        )

        max_games = TRAIN_PARAMS.max_train_games + TRAIN_PARAMS.max_val_games
        probe_data = construct_linear_probe_data(
            input_dataframe_file,
            layer,
            dataset_prefix,
            n_layers,
            model_name,
            config,
            max_games,
            DEVICE,
        )

        logging_dict = init_logging_dict(
            config, probe_data, split, dataset_prefix, model_name, n_layers
        )

        train_linear_probe_cross_entropy(probe_data, config, logging_dict)
