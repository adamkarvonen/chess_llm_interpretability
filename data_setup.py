# For data setup
import chess
import numpy as np
import pandas as pd
import pickle

DATA_DIR = "data/"

### BEGIN DATA PREPARATION ###
# This takes a csv file and creates the following files:
# - {split}board_seqs_string.csv (basically a copy of the csv but with the header removed
# and an assert statement to ensure that all rows are of the same length)
# - {split}board_seqs_int.npy board_seqs_string.csv but with the strings converted to integer sequences using the meta.pkl file
# - {split}dots_indices.npy a NumPy array of indices of every '.' in the string. This will hopefully provide a reasonable starting point for training a linear probe.
# This was developed in chess_utils.ipynb. I just copy pasted the relevant cells into here for convenience. Sorry for the messiness.


def find_dots_indices(moves_string: str) -> list[int]:
    """Returns a NumPy array of indices of every '.' in the string.
    This will hopefully provide a reasonable starting point for training a linear probe.
    """
    indices = [index for index, char in enumerate(moves_string) if char == "."]
    return indices


# meta is used to encode the string pgn strings into integer sequences
with open("nanogpt/out/meta.pkl", "rb") as f:
    meta = pickle.load(f)

print(meta)

stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print(encode(";1.e4 "))
print(decode(encode(";1.e4 ")))


def process_csv(csv_filename: str):
    prefix = ""
    split = "train_"
    if "test" in csv_filename:
        split = "test_"

    if "skill" in csv_filename:
        prefix = "skill_"

    df = pd.read_csv(f"{DATA_DIR}{csv_filename}")
    row_length = len(df["transcript"].iloc[0])
    num_games = len(df)

    assert all(
        df["transcript"].apply(lambda x: len(x) == row_length)
    ), "Not all transcripts are of length {}".format(row_length)

    df["transcript"].to_csv(
        f"{DATA_DIR}{prefix}{split}board_seqs_str.csv", index=False, header=False
    )

    print(len(df), len(df["transcript"].iloc[0]))

    assert (len(df), len(df["transcript"].iloc[0])) == (num_games, row_length)

    df = pd.read_csv(f"{DATA_DIR}{csv_filename}")
    encoded_df = df["transcript"].apply(encode)
    print(encoded_df.head())
    board_seqs_int = np.array(encoded_df.apply(list).tolist())
    print(board_seqs_int.shape)
    assert board_seqs_int.shape == (num_games, row_length)

    np.save(f"{DATA_DIR}{prefix}{split}board_seqs_int.npy", board_seqs_int)

    df = pd.read_csv(f"{DATA_DIR}{csv_filename}")
    dots_indices_series = df["transcript"].apply(find_dots_indices)

    shortest_length = dots_indices_series.apply(len).min()
    print("Shortest length:", shortest_length)

    dots_indices_series = dots_indices_series.apply(lambda x: x[:shortest_length])
    assert all(
        len(lst) == shortest_length for lst in dots_indices_series
    ), "Not all lists have the same length"

    print(dots_indices_series.head())
    dots_indices = np.array(dots_indices_series.apply(list).tolist())
    print(dots_indices.shape)
    assert dots_indices.shape == (num_games, shortest_length)
    np.save(f"{DATA_DIR}{prefix}{split}dots_indices.npy", dots_indices)

    if prefix == "skill_":
        df = pd.read_csv(f"{DATA_DIR}{csv_filename}")
        # Extract skill levels as integers
        skill_levels_list = [int(x.split()[1]) for x in df["player_two"]]

        # Convert the list to a numpy array
        skill_level = np.array(skill_levels_list)
        print(skill_level.shape)
        assert skill_level.shape == (num_games,)
        print("Skill level shape:", skill_level.shape)
        np.save(f"{DATA_DIR}{prefix}{split}skill_level.npy", skill_level)


# process_csv("train.csv")
# process_csv("test.csv")
process_csv("skill_train.csv")
process_csv("skill_test.csv")
### END DATA PREPARATION ###
