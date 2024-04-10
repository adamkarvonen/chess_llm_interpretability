import sys
import os

# Not the ideal way of doing things, but it works. This way all test functions can pull models / probes / data from the expected location
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import caa
import train_test_chess
import chess_utils
import torch

MAXIMUM_TRAINING_GAMES = 500
DATA_DIR = "data/"

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


# Just a smoke test
def test_caa():

    config = chess_utils.skill_config

    caa_type = "simple"

    previous_layer_activations = {}

    dataset_prefix = "lichess_"
    layer = 5
    split = "train"
    n_layers = 16
    model_name = f"tf_lens_{dataset_prefix}{n_layers}layers_ckpt_no_optimizer"
    config.levels_of_interest = [0, 5]
    input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
    config = chess_utils.set_config_min_max_vals_and_column_name(
        config, input_dataframe_file, dataset_prefix
    )
    config.pos_start = 25

    probe_data = train_test_chess.construct_linear_probe_data(
        input_dataframe_file,
        dataset_prefix,
        n_layers,
        model_name,
        config,
        MAXIMUM_TRAINING_GAMES,
        device,
    )

    cascade_layers = ""

    activation_name = f"TEST_ONLY_type=caa_{caa_type}{cascade_layers}_model={n_layers}layers_layer={layer}_activations"

    logging_dict = train_test_chess.init_logging_dict(
        layer,
        config,
        split,
        dataset_prefix,
        model_name,
        n_layers,
        train_test_chess.TRAIN_PARAMS,
    )

    previous_layer_activations[layer] = caa.create_contrastive_activations(
        activation_name, probe_data, config, logging_dict, layer, MAXIMUM_TRAINING_GAMES
    )
