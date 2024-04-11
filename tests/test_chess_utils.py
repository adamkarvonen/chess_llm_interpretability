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
    test3 = ";1."

    ans1 = [[0, 1, 2, 3, 4], [8, 9, 10, 11, 12, 13], [17, 18]]
    ans2 = [[0, 1, 2, 3, 4], [8, 9, 10, 11, 12, 13]]
    ans3 = [[0, 1, 2]]

    assert chess_utils.get_all_white_pos_indices(test1) == ans1
    assert chess_utils.get_all_white_pos_indices(test2) == ans2
    assert chess_utils.get_all_white_pos_indices(test3) == ans3


def test_black_pos_indices():
    test1 = ";1.e4 c5 2.Nf3 d6 3"
    test2 = ";1.e4 c5 2.Nf3 d"
    test3 = ";1."

    ans1 = [[5, 6, 7], [14, 15, 16]]
    ans2 = [[5, 6, 7], [14, 15]]
    ans3 = []

    assert chess_utils.get_all_black_pos_indices(test1) == ans1
    assert chess_utils.get_all_black_pos_indices(test2) == ans2
    assert chess_utils.get_all_black_pos_indices(test3) == ans3


def test_board_to_piece_state():

    test_str = ";1.e4 e5 2.Nf3"
    board = chess_utils.pgn_string_to_board(test_str)
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


def test_white_piece_prev_pos_indices():
    test1 = ";1.e4 e5 2.Nf3 Nc6 3."

    board = chess_utils.pgn_string_to_board(test1)
    move_san = board.parse_san("d4")
    prev_pos_indices = chess_utils.get_all_white_piece_prev_pos_indices(test1, board, move_san)
    # d2 pawn has been there since the start of the game
    expected_ans = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 18, 19, 20]
    assert prev_pos_indices == expected_ans

    board = chess_utils.pgn_string_to_board(test1)
    move_san = board.parse_san("Nh4")
    prev_pos_indices = chess_utils.get_all_white_piece_prev_pos_indices(test1, board, move_san)

    # Nf3 knight moved there on the last move. NOTE: We also want to erase the piece during the move where it was placed
    expected_ans = [8, 9, 10, 11, 12, 13, 18, 19, 20]
    assert prev_pos_indices == expected_ans

    test2 = ";1."
    board = chess_utils.pgn_string_to_board(test2)
    move_san = board.parse_san("e4")
    prev_pos_indices = chess_utils.get_all_white_piece_prev_pos_indices(test2, board, move_san)
    expected_ans = [0, 1, 2]
    assert prev_pos_indices == expected_ans


def test_black_piece_prev_pos_indices():
    test1 = ";1.e4 e5 2.Nf3 Nc6 3."

    board = chess_utils.pgn_string_to_board(test1)
    move_san = board.parse_san("d4")
    prev_pos_indices = chess_utils.get_all_black_piece_prev_pos_indices(test1, board, move_san)
    # d2 pawn has been there since the start of the game
    expected_ans = [5, 6, 7, 14, 15, 16, 17]
    assert prev_pos_indices == expected_ans

    board = chess_utils.pgn_string_to_board(test1)
    move_san = board.parse_san("Nh4")
    prev_pos_indices = chess_utils.get_all_black_piece_prev_pos_indices(test1, board, move_san)

    # Nf3 knight moved there on the last move. NOTE: We also want to erase the piece during the move where it was placed
    expected_ans = [14, 15, 16, 17]
    assert prev_pos_indices == expected_ans

    test2 = ";1."
    board = chess_utils.pgn_string_to_board(test2)
    move_san = board.parse_san("e4")
    prev_pos_indices = chess_utils.get_all_black_piece_prev_pos_indices(test2, board, move_san)
    expected_ans = []
    assert prev_pos_indices == expected_ans
