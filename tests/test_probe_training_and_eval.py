import sys
import os

# Not the ideal way of doing things, but it works. This way all test functions can pull models / probes / data from the expected location
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import train_test_chess
import torch
import chess_utils

DATA_DIR = "data/"

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

TRAIN_PARAMS = train_test_chess.TrainingParams()
TRAIN_PARAMS.max_iters = 10000
TRAIN_PARAMS.num_epochs = 2
TRAIN_PARAMS.max_train_games = 5000
TRAIN_PARAMS.max_test_games = 2000

# TRAIN_EPSILON is fairly large because the probes haven't converged and are trained on random permutations of the training data
TRAIN_EPSILON = 0.03
TEST_EPSILON = 0.002
# IMPORTANT: You must train probes on this model or the asserts will fail
# https://huggingface.co/adamkarvonen/chess_llms/blob/main/lichess_8layers_ckpt_no_optimizer.pt
# At a small epsilon and only 10k iters, the probes haven't converged so it will be sensitive to changes in the model
# But at least this way testing the training code only takes 10 minutes


def test_piece_train_linear_probe_cross_entropy():
    torch.set_grad_enabled(True)
    config = chess_utils.piece_config
    first_layer = 0
    last_layer = 7

    dataset_prefix = "lichess_"
    split = "train"
    n_layers = 8
    model_name = f"tf_lens_{dataset_prefix}{n_layers}layers_ckpt_no_optimizer"

    input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
    config = chess_utils.set_config_min_max_vals_and_column_name(
        config, input_dataframe_file, dataset_prefix
    )

    max_games = TRAIN_PARAMS.max_train_games + TRAIN_PARAMS.max_val_games

    probe_data = train_test_chess.construct_linear_probe_data(
        input_dataframe_file,
        dataset_prefix,
        n_layers,
        model_name,
        config,
        max_games,
        DEVICE,
    )

    probes = train_test_chess.populate_probes_dict(
        list(range(first_layer, last_layer + 1)),
        config,
        TRAIN_PARAMS,
        split,
        dataset_prefix,
        model_name,
        n_layers,
    )

    for probe in probes:
        probes[probe].probe_name = probes[probe].probe_name.replace("tf_lens", "TEST_ONLY_tf_lens")

    final_accs = train_test_chess.train_linear_probe_cross_entropy(
        probes, probe_data, config, TRAIN_PARAMS
    )

    expected_final_accs = {
        0: 0.745,
        1: 0.746,
        2: 0.768,
        3: 0.803,
        4: 0.863,
        5: 0.981,
        6: 0.978,
        7: 0.961,
    }

    for layer in range(first_layer, last_layer + 1):
        assert (
            abs(final_accs[layer] - expected_final_accs[layer]) < TRAIN_EPSILON
        ), f"Accuracy mismatch for layer {layer}"


def test_skill_train_linear_probe_cross_entropy():
    torch.set_grad_enabled(True)
    config = chess_utils.skill_config
    first_layer = 0
    last_layer = 7

    dataset_prefix = "lichess_"
    split = "train"
    n_layers = 8
    model_name = f"tf_lens_{dataset_prefix}{n_layers}layers_ckpt_no_optimizer"

    input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"
    config = chess_utils.set_config_min_max_vals_and_column_name(
        config, input_dataframe_file, dataset_prefix
    )

    max_games = TRAIN_PARAMS.max_train_games + TRAIN_PARAMS.max_val_games

    probe_data = train_test_chess.construct_linear_probe_data(
        input_dataframe_file,
        dataset_prefix,
        n_layers,
        model_name,
        config,
        max_games,
        DEVICE,
    )

    probes = train_test_chess.populate_probes_dict(
        list(range(first_layer, last_layer + 1)),
        config,
        TRAIN_PARAMS,
        split,
        dataset_prefix,
        model_name,
        n_layers,
    )

    for probe in probes:
        probes[probe].probe_name = probes[probe].probe_name.replace("tf_lens", "TEST_ONLY_tf_lens")

    final_accs = train_test_chess.train_linear_probe_cross_entropy(
        probes, probe_data, config, TRAIN_PARAMS
    )

    expected_final_accs = {
        0: 0.645,
        1: 0.663,
        2: 0.641,
        3: 0.709,
        4: 0.820,
        5: 0.838,
        6: 0.876,
        7: 0.878,
    }

    for layer in range(first_layer, last_layer + 1):
        assert (
            abs(final_accs[layer] - expected_final_accs[layer]) < TRAIN_EPSILON
        ), f"Accuracy mismatch for layer {layer}"


def test_linear_probe_cross_entropy_test():

    expected_results = {
        "tf_lens_lichess_8layers_ckpt_no_optimizer_chess_piece_probe_layer_5.pth": 0.9907,
        "tf_lens_lichess_8layers_ckpt_no_optimizer_chess_skill_probe_layer_7.pth": 0.8856,
    }

    print(expected_results)
    for probe_to_test in expected_results.keys():
        probe_file_location = f"{train_test_chess.SAVED_PROBE_DIR}{probe_to_test}"
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

            probe_data = train_test_chess.construct_linear_probe_data(
                input_dataframe_file,
                dataset_prefix,
                n_layers,
                model_name,
                config,
                TRAIN_PARAMS.max_test_games,
                DEVICE,
            )

            logging_dict = train_test_chess.init_logging_dict(
                layer, config, split, dataset_prefix, model_name, n_layers, TRAIN_PARAMS
            )

            result = train_test_chess.test_linear_probe_cross_entropy(
                probe_file_location, probe_data, config, logging_dict, TRAIN_PARAMS
            )

            assert (
                abs(result - expected_results[probe_to_test]) < TEST_EPSILON
            ), f"Accuracy mismatch for probe {probe_to_test}"
