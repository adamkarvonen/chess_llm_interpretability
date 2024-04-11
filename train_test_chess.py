from transformer_lens import HookedTransformer, HookedTransformerConfig
import einops
import torch
from tqdm import tqdm
from fancy_einsum import einsum
from dataclasses import dataclass, field
import pandas as pd
import pickle
import logging
from typing import Optional
from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
from beartype import beartype
import collections
import chess_utils
from chess_utils import PlayerColor, Config
import argparse

logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)

# Add handler to this logger if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

MODEL_DIR = "models/"
DATA_DIR = "data/"
PROBE_DIR = "linear_probes/"
SAVED_PROBE_DIR = "linear_probes/saved_probes/"
WANDB_PROJECT = "chess_linear_probes"
BATCH_SIZE = 2
D_MODEL = 512
N_HEADS = 8
WANDB_LOGGING = False

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
logger.info(f"Using device: {DEVICE}")

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
class SingleProbe:
    linear_probe: torch.Tensor
    probe_name: str
    optimiser: torch.optim.AdamW
    logging_dict: dict
    loss: torch.Tensor = torch.tensor(0.0)
    accuracy: torch.Tensor = torch.tensor(0.0)
    accuracy_queue: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=1000)
    )


@dataclass
class LinearProbeData:
    model: HookedTransformer
    custom_indices: torch.Tensor
    board_seqs_int: torch.Tensor
    board_seqs_string: list[str]
    skill_stack: torch.Tensor
    user_state_dict_one_hot_mapping: Optional[dict[int, int]] = None


def get_transformer_lens_model(
    model_name: str, n_layers: int, device: torch.device
) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=D_MODEL,
        d_head=int(D_MODEL / N_HEADS),
        n_heads=N_HEADS,
        d_mlp=D_MODEL * 4,
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
    layer: int,
    config: Config,
    split: str,
    dataset_prefix: str,
    model_name: str,
    n_layers: int,
    train_params: TrainingParams,
) -> dict:

    indexing_function_name = config.custom_indexing_function.__name__

    wandb_run_name = f"{config.linear_probe_name}_{model_name}_layer_{layer}_indexing_{indexing_function_name}_max_games_{train_params.max_train_games}"
    if config.levels_of_interest is not None:
        wandb_run_name += "_levels"
        for level in config.levels_of_interest:
            wandb_run_name += f"_{level}"

    logging_dict = {
        "linear_probe_name": config.linear_probe_name,
        "layer": layer,
        "indexing_function_name": indexing_function_name,
        "batch_size": BATCH_SIZE,
        "lr": train_params.lr,
        "wd": train_params.wd,
        "pos_start": config.pos_start,
        "num_epochs": train_params.num_epochs,
        "num_games": train_params.max_train_games,
        "modes": train_params.modes,
        "wandb_project": WANDB_PROJECT,
        "config_name": config.linear_probe_name,
        "column_name": config.column_name,
        "levels_of_interest": config.levels_of_interest,
        "split": split,
        "dataset_prefix": dataset_prefix,
        "model_name": model_name,
        "n_layers": n_layers,
        "wandb_run_name": wandb_run_name,
        "player_color": config.player_color.value,
    }

    return logging_dict


def get_board_seqs_string(df: pd.DataFrame) -> list[str]:
    row_length = len(df["transcript"].iloc[0])

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
    logger.info(f"custom_indices shape: {custom_indices.shape}")
    return custom_indices


@jaxtyped(typechecker=beartype)
def prepare_data_batch(
    indices: Int[Tensor, "batch_size"],
    probe_data: LinearProbeData,
    config: Config,
    layers: list[int],
) -> tuple[
    Int[Tensor, "modes batch_size num_white_moves num_rows num_cols num_options"],
    dict[int, Float[Tensor, "batch_size num_white_moves d_model"]],
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

    resid_post_dict = {}

    with torch.inference_mode():
        _, cache = probe_data.model.run_with_cache(games_int.to(DEVICE)[:, :-1], return_type=None)
        for layer in layers:
            resid_post_dict[layer] = cache["resid_post", layer][
                :, :
            ]  # shape (batch_size, pgn_str_length - 1, d_model)

    # Not the most efficient way to do this, but it's clear and readable
    for layer in layers:
        resid_post = resid_post_dict[layer]
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
        resid_post_dict[layer] = torch.stack(
            indexed_resid_posts
        )  # shape (batch_size, num_white_moves, d_model)

    return state_stack_one_hot, resid_post_dict


def populate_probes_dict(
    layers: list[int],
    config: Config,
    train_params: TrainingParams,
    split,
    dataset_prefix,
    model_name,
    n_layers,
) -> dict[int, SingleProbe]:
    probes = {}
    for layer in layers:
        logging_dict = init_logging_dict(
            layer, config, split, dataset_prefix, model_name, n_layers, TRAIN_PARAMS
        )
        linear_probe_name = (
            f"{PROBE_DIR}{logging_dict['model_name']}_{config.linear_probe_name}_layer_{layer}.pth"
        )
        linear_probe = torch.randn(
            train_params.modes,
            D_MODEL,
            config.num_rows,
            config.num_cols,
            get_one_hot_range(config),
            requires_grad=False,
            device=DEVICE,
        ) / torch.sqrt(torch.tensor(D_MODEL))
        linear_probe.requires_grad = True
        logger.info(f"linear_probe shape: {linear_probe.shape}")

        optimiser = torch.optim.AdamW(
            [linear_probe],
            lr=train_params.lr,
            betas=(train_params.beta1, train_params.beta2),
            weight_decay=train_params.wd,
        )
        probes[layer] = SingleProbe(
            linear_probe=linear_probe,
            probe_name=linear_probe_name,
            optimiser=optimiser,
            logging_dict=logging_dict,
        )
    return probes


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
# You could disable this if using a training set with over 5k games, as there shouldn't be any overfitting
@torch.no_grad()
def estimate_loss(
    train_games: int,
    val_games: int,
    probes: dict[int, SingleProbe],
    probe_data: LinearProbeData,
    config: Config,
    one_hot_range: int,
    layers: list[int],
    train_params: TrainingParams,
) -> dict[int, dict[str, dict[str, float]]]:
    out = {}

    for layer in probes:
        out[layer] = {
            "train": {"loss": 0.0, "accuracy": 0.0},
            "val": {"loss": 0.0, "accuracy": 0.0},
        }

    eval_iters = (train_params.eval_iters // BATCH_SIZE) * BATCH_SIZE

    train_indices = torch.randperm(train_games)[:eval_iters]
    val_indices = torch.randperm(val_games) + train_games  # to avoid overlap
    val_indices = val_indices[:eval_iters]

    split_indices = {"train": train_indices, "val": val_indices}
    for split in split_indices:
        losses: dict[int, list[float]] = {}
        accuracies: dict[int, list[float]] = {}
        for layer in probes:
            losses[layer] = []
            accuracies[layer] = []
        for k in range(0, eval_iters, BATCH_SIZE):
            indices = split_indices[split][k : k + BATCH_SIZE]

            state_stack_one_hot, resid_post_dict = prepare_data_batch(
                indices, probe_data, config, layers
            )

            for layer in probes:
                loss, accuracy = linear_probe_forward_pass(
                    probes[layer].linear_probe,
                    state_stack_one_hot,
                    resid_post_dict[layer],
                    one_hot_range,
                )
                losses[layer].append(loss.item())
                accuracies[layer].append(accuracy.item())
        for layer in layers:
            out[layer][split]["loss"] = sum(losses[layer]) / len(losses[layer])
            out[layer][split]["accuracy"] = sum(accuracies[layer]) / len(accuracies[layer])
    return out


def train_linear_probe_cross_entropy(
    probes: dict[int, SingleProbe],
    probe_data: LinearProbeData,
    config: Config,
    train_params: TrainingParams,
) -> dict[int, float]:
    """Trains a linear probe on the train set, contained in probe_data. Saves all probes to disk.
    Returns a dict of layer: final avg_acc over the last 1,000 iterations.
    This dict is also used as an end to end test for the function."""

    first_layer = min(probes.keys())
    layers = list(probes.keys())
    all_layers_str = "_".join([str(layer) for layer in layers])
    assert probes[first_layer].logging_dict["split"] == "train", "Don't train on the test set"

    val_games = (train_params.max_val_games // BATCH_SIZE) * BATCH_SIZE
    train_games = (train_params.max_train_games // BATCH_SIZE) * BATCH_SIZE

    num_games = train_games + val_games

    if len(probe_data.board_seqs_int) < num_games:
        raise ValueError("Not enough games to train on")
        # We raise an error so it doesn't fail silently. If we want to use less games, we can comment the error out
        # and add some logic to set train and val games to the number of games we have

    one_hot_range = get_one_hot_range(config)

    if WANDB_LOGGING:
        import wandb

        wandb.init(
            project=WANDB_PROJECT,
            name=f"layers:{all_layers_str}_" + probes[first_layer].logging_dict["wandb_run_name"],
            config=probes[first_layer].logging_dict,
        )

    current_iter = 0
    for epoch in range(train_params.num_epochs):
        full_train_indices = torch.randperm(train_games)
        for i in tqdm(range(0, train_games, BATCH_SIZE)):

            indices = full_train_indices[i : i + BATCH_SIZE]  # shape batch_size

            state_stack_one_hot, resid_post_dict = prepare_data_batch(
                indices, probe_data, config, layers
            )

            for layer in probes:

                probes[layer].loss, probes[layer].accuracy = linear_probe_forward_pass(
                    probes[layer].linear_probe,
                    state_stack_one_hot,
                    resid_post_dict[layer],
                    one_hot_range,
                )

                probes[layer].loss.backward()
                probes[layer].optimiser.step()
                probes[layer].optimiser.zero_grad()

                probes[layer].accuracy_queue.append(probes[layer].accuracy.item())

            if i % 100 == 0:
                if WANDB_LOGGING:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "iter": current_iter,
                        }
                    )
                for layer in probes:
                    avg_acc = sum(probes[layer].accuracy_queue) / len(probes[layer].accuracy_queue)
                    logger.info(
                        f"epoch {epoch}, iter {i}, layer {layer}, acc {probes[layer].accuracy:.3f}, loss {probes[layer].loss:.3f}, avg acc {avg_acc:.3f}"
                    )
                    if WANDB_LOGGING:
                        wandb.log(
                            {
                                f"layer_{layer}_loss": probes[layer].loss,
                                f"layer_{layer}_acc": probes[layer].accuracy,
                                f"layer_{layer}_avg_acc": avg_acc,
                            }
                        )

            if current_iter % 1000 == 0:
                losses = estimate_loss(
                    train_games,
                    val_games,
                    probes,
                    probe_data,
                    config,
                    one_hot_range,
                    layers,
                    train_params,
                )
                for layer in probes:
                    logger.info(
                        f"epoch {epoch}, layer {layer}, train loss: {losses[layer]['train']['loss']:.3f}, val loss: {losses[layer]['val']['loss']:.3f}, train acc: {losses[layer]['train']['accuracy']:.3f}, val acc: {losses[layer]['val']['accuracy']:.3f}"
                    )
                    if WANDB_LOGGING:
                        wandb.log(
                            {
                                f"layer_{layer}_train_loss": losses[layer]["train"]["loss"],
                                f"layer_{layer}_train_acc": losses[layer]["train"]["accuracy"],
                                f"layer_{layer}_val_loss": losses[layer]["val"]["loss"],
                                f"layer_{layer}_val_acc": losses[layer]["val"]["accuracy"],
                            }
                        )
            current_iter += BATCH_SIZE
    final_accs = {}
    for layer in probes:
        checkpoint = {
            "linear_probe": probes[layer].linear_probe,
            "final_loss": probes[layer].loss,
            "iters": current_iter,
            "epochs": epoch,
            "acc": probes[layer].accuracy,
        }
        # Update the checkpoint dictionary with the contents of logging_dict
        checkpoint.update(probes[layer].logging_dict)
        torch.save(checkpoint, probes[layer].probe_name)
        final_accs[layer] = sum(probes[layer].accuracy_queue) / len(probes[layer].accuracy_queue)
        logger.info(f"layer {layer}, final acc: {final_accs[layer]}")
    return final_accs


def construct_linear_probe_data(
    input_dataframe_file: str,
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
        model=model,
        custom_indices=custom_indices,
        board_seqs_int=board_seqs_int,
        board_seqs_string=board_seqs_string,
        skill_stack=skill_stack,
        user_state_dict_one_hot_mapping=user_state_dict_one_hot_mapping,
    )

    return probe_data


def get_one_hot_range(config: Config) -> int:
    one_hot_range = config.max_val - config.min_val + 1
    if config.levels_of_interest is not None:
        one_hot_range = len(config.levels_of_interest)
    return one_hot_range


@torch.no_grad()
def test_linear_probe_cross_entropy(
    linear_probe_name: str,
    probe_data: LinearProbeData,
    config: Config,
    logging_dict: dict,
    train_params: TrainingParams,
) -> float:
    """Takes a linear probe and tests it on the test set, contained in probe_data. Saves the results to a pickle file.
    Returns a float representing the average accuracy of the probe on the test set. This is also used as an end to end test for the function.
    """
    assert logging_dict["split"] == "test", "Don't test on the train set"

    num_games = (train_params.max_test_games // BATCH_SIZE) * BATCH_SIZE

    if (len(probe_data.board_seqs_int) // BATCH_SIZE) * BATCH_SIZE < num_games:
        raise ValueError("Not enough games to test on")
        # We raise an error so it doesn't fail silently. If we want to use less games, we can comment the error out
        num_games = (len(probe_data.board_seqs_int) // BATCH_SIZE) * BATCH_SIZE

    one_hot_range = get_one_hot_range(config)

    logging_dict["num_games"] = num_games

    checkpoint = torch.load(linear_probe_name, map_location=DEVICE)
    linear_probe = checkpoint["linear_probe"]
    logger.info(f"linear_probe shape: {linear_probe.shape}")
    logger.info(f"custom_indices shape: {probe_data.custom_indices.shape}")

    layer = logging_dict["layer"]

    current_iter = 0
    accuracy_list = []
    loss_list = []
    full_test_indices = torch.arange(0, num_games)
    for i in tqdm(range(0, num_games, BATCH_SIZE)):
        indices = full_test_indices[i : i + BATCH_SIZE]  # shape batch_size

        state_stack_one_hot, resid_post_dict = prepare_data_batch(
            indices, probe_data, config, [layer]
        )

        loss, accuracy = linear_probe_forward_pass(
            linear_probe, state_stack_one_hot, resid_post_dict[layer], one_hot_range
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

    return average_accuracy


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train or test chess probes on piece or skill data."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "train"],
        default="train",
        help='Mode to run the script in: "test" or "train".',
    )
    parser.add_argument(
        "--probe",
        type=str,
        choices=["piece", "skill"],
        default="piece",
        help='Type of probe to use: "piece" for piece board state or "skill" for player skill level.',
    )
    parser.add_argument(
        "--wandb_logging",
        action="store_true",
        help="Enable logging to Weights & Biases. Default is False.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    WANDB_LOGGING = args.wandb_logging
    if args.mode == "test":
        # saved_probes = [
        #     file
        #     for file in os.listdir(SAVED_PROBE_DIR)
        #     if os.path.isfile(os.path.join(SAVED_PROBE_DIR, file))
        # ]
        saved_probes = []

        # Quick and janky way to select between piece and skill probes
        if args.probe == "piece":
            saved_probes = [
                "tf_lens_lichess_8layers_ckpt_no_optimizer_chess_piece_probe_layer_5.pth"
            ]
        elif args.probe == "skill":
            saved_probes = [
                "tf_lens_lichess_8layers_ckpt_no_optimizer_chess_skill_probe_layer_5.pth"
            ]

        print(saved_probes)

        # NOTE: This is very inefficient. The expensive part is forwarding the GPT, which we should only have to do once.
        # With little effort, we could test probes on all layers at once. This would be much faster.
        # But, I can test the probes in 20 minutes and it was a one-off thing, so I didn't bother.
        # My strategy for development / hyperparameter testing was to iterate on the train side, then do the final test on the test side.
        # As long as you have a reasonable training dataset size, you should be able to get a good idea of final test accuracy
        # by looking at the training accuracy after a few epochs.
        for probe_to_test in saved_probes:
            probe_file_location = f"{SAVED_PROBE_DIR}{probe_to_test}"
            # We will populate all parameters using information in the probe state dict
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
                split = "test"

                input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
                config = chess_utils.set_config_min_max_vals_and_column_name(
                    config, input_dataframe_file, dataset_prefix
                )

                probe_data = construct_linear_probe_data(
                    input_dataframe_file,
                    dataset_prefix,
                    n_layers,
                    model_name,
                    config,
                    TRAIN_PARAMS.max_test_games,
                    DEVICE,
                )

                logging_dict = init_logging_dict(
                    layer, config, split, dataset_prefix, model_name, n_layers, TRAIN_PARAMS
                )

                test_linear_probe_cross_entropy(
                    probe_file_location, probe_data, config, logging_dict, TRAIN_PARAMS
                )
    elif args.mode == "train":
        config = chess_utils.piece_config
        if args.probe == "skill":
            config = chess_utils.skill_config

        player_color = PlayerColor.WHITE
        first_layer = 0
        last_layer = 7

        # When training a probe, you have to set all parameters such as model name, dataset prefix, etc.
        dataset_prefix = "lichess_"
        split = "train"
        n_layers = 8
        model_name = f"tf_lens_{dataset_prefix}{n_layers}layers_ckpt_no_optimizer"

        input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
        config = chess_utils.set_config_min_max_vals_and_column_name(
            config, input_dataframe_file, dataset_prefix
        )
        config = chess_utils.update_config_using_player_color(player_color, config)

        max_games = TRAIN_PARAMS.max_train_games + TRAIN_PARAMS.max_val_games
        probe_data = construct_linear_probe_data(
            input_dataframe_file,
            dataset_prefix,
            n_layers,
            model_name,
            config,
            max_games,
            DEVICE,
        )

        probes = populate_probes_dict(
            list(range(first_layer, last_layer + 1)),
            config,
            TRAIN_PARAMS,
            split,
            dataset_prefix,
            model_name,
            n_layers,
        )

        train_linear_probe_cross_entropy(probes, probe_data, config, TRAIN_PARAMS)
