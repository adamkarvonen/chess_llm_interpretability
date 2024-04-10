import sys
import os

# Not the ideal way of doing things, but it works. This way all test functions can pull models / probes / data from the expected location
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import board_state_interventions

# TOLERANCE is fairly high because we are testing on a small amount of games for efficiency, won't converge to the expected value
# This is more of a smoke test than anything
TOLERANCE = 0.2


def test_single_layer_interventions():
    intervention_type = board_state_interventions.InterventionType.SINGLE_SCALE

    num_games = 2

    probe_names = {}
    first_layer = 5
    last_layer = 5
    GPT_LAYER_COUNT = 8

    for i in range(first_layer, last_layer + 1, 1):
        probe_names[i] = (
            f"tf_lens_lichess_{GPT_LAYER_COUNT}layers_ckpt_no_optimizer_chess_piece_probe_layer_{i}.pth"
        )
    probe_data = board_state_interventions.get_probe_data(probe_names[first_layer], num_games)

    piece_coe = 1.0
    blank_coe = 0.0

    scales = [3.0]

    recording_name = f"TEST_ONLY_n_layers={GPT_LAYER_COUNT}_intervention_type={intervention_type.value}_first_layer={first_layer}_last_layer={last_layer}_p={piece_coe}_b={blank_coe}_scales="
    for scale in scales:
        recording_name += f"{str(scale).replace('.', '')[:5]}_"

    print(f"Recording name: {recording_name}")

    success_rate = board_state_interventions.perform_board_interventions(
        probe_names,
        probe_data,
        num_games,
        intervention_type,
        recording_name,
        track_outputs=False,
        scales=scales,
    )

    expected_success_rate = 0.8

    assert abs(success_rate - expected_success_rate) < TOLERANCE, f"Success rate mismatch"
