import sys
import os

# Not the ideal way of doing things, but it works. This way all test functions can pull models / probes / data from the expected location
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import chess_utils
import chess
import torch


def test_white_pos_indices():
    test1 = ";1.e4 c5 2.Nf3 d6 3"
    test2 = ";1.e4 c5 2.Nf3 d"

    ans1 = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 17, 18]
    ans2 = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13]

    assert chess_utils.get_all_white_pos_indices(test1) == ans1
    assert chess_utils.get_all_white_pos_indices(test2) == ans2


def test_black_pos_indices():
    test1 = ";1.e4 c5 2.Nf3 d6 3"
    test2 = ";1.e4 c5 2.Nf3 d"

    ans1 = [5, 6, 7, 14, 15, 16]
    ans2 = [5, 6, 7, 14, 15]

    assert chess_utils.get_all_black_pos_indices(test1) == ans1
    assert chess_utils.get_all_black_pos_indices(test2) == ans2


def test_board_to_piece_state():

    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    state = chess_utils.board_to_piece_state(board)

    expected_state = torch.tensor(
        [
            [4, 2, 3, 5, 6, 3, 0, 4],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, 0, -1, -1, -1],
            [-4, -2, -3, -5, -6, -3, -2, -4],
        ],
        dtype=torch.int,
    )

    assert torch.equal(state, expected_state)
