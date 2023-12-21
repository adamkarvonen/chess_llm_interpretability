# For data setup
import chess
import numpy as np
import pandas as pd
import pickle

# For nanogpt to transformer lens conversion
import torch
import einops

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

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
    split = "train_"
    if "test" in csv_filename:
        split = "test_"

    df = pd.read_csv(f"{DATA_DIR}{csv_filename}")
    row_length = len(df["transcript"].iloc[0])
    num_games = len(df)

    assert all(
        df["transcript"].apply(lambda x: len(x) == row_length)
    ), "Not all transcripts are of length {}".format(row_length)

    df["transcript"].to_csv(
        f"{DATA_DIR}{split}board_seqs_string.csv", index=False, header=False
    )

    print(len(df), len(df["transcript"].iloc[0]))

    assert (len(df), len(df["transcript"].iloc[0])) == (num_games, row_length)

    df = pd.read_csv(f"{DATA_DIR}{csv_filename}")
    encoded_df = df["transcript"].apply(encode)
    print(encoded_df.head())
    board_seqs_int = np.array(encoded_df.apply(list).tolist())
    print(board_seqs_int.shape)
    assert board_seqs_int.shape == (num_games, row_length)

    np.save(f"{DATA_DIR}{split}board_seqs_int.npy", board_seqs_int)

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
    np.save(f"{DATA_DIR}{split}dots_indices.npy", dots_indices)


process_csv("train.csv")
process_csv("test.csv")
### END DATA PREPARATION ###

### BEGIN MODEL SETUP ###
# Our pytorch model is in the nanogpt format. For easy linear probing of the residual stream, we want to convert
# it to the transformer lens format. This is done in the following code block.
# This code was developed using Neel Nanda's othello_reference/Othello_GPT.ipynb as a reference.
# Once again, I just copy pasted the relevant cells into here for convenience. Sorry for the messiness.

torch.set_grad_enabled(False)

LOAD_AND_CONVERT_CHECKPOINT = True

device = "cpu"

MODEL_DIR = "models/"

checkpoint = torch.load(f"{MODEL_DIR}ckpt_3487k_iters_pre_dropout_no_optim.pt")

# Print the keys of the checkpoint dictionary
print(checkpoint.keys())
model_state = checkpoint["model"]
# for key, value in model_state.items():
#     print(key, value.shape)


def convert_to_transformer_lens_format(in_sd, n_layers, n_heads):
    out_sd = {}
    out_sd["pos_embed.W_pos"] = in_sd["_orig_mod.transformer.wpe.weight"]
    out_sd["embed.W_E"] = in_sd["_orig_mod.transformer.wte.weight"]

    out_sd["ln_final.w"] = in_sd["_orig_mod.transformer.ln_f.weight"]
    out_sd["ln_final.b"] = torch.zeros_like(in_sd["_orig_mod.transformer.ln_f.weight"])
    out_sd["unembed.W_U"] = in_sd["_orig_mod.lm_head.weight"].T

    for layer in range(n_layers):
        layer_key = f"_orig_mod.transformer.h.{layer}"

        # Layer Norms
        out_sd[f"blocks.{layer}.ln1.w"] = in_sd[f"{layer_key}.ln_1.weight"]
        out_sd[f"blocks.{layer}.ln1.b"] = torch.zeros_like(
            in_sd[f"{layer_key}.ln_1.weight"]
        )
        out_sd[f"blocks.{layer}.ln2.w"] = in_sd[f"{layer_key}.ln_2.weight"]
        out_sd[f"blocks.{layer}.ln2.b"] = torch.zeros_like(
            in_sd[f"{layer_key}.ln_2.weight"]
        )

        W = in_sd[f"{layer_key}.attn.c_attn.weight"]
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=0)
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        out_sd[f"blocks.{layer}.attn.W_Q"] = W_Q
        out_sd[f"blocks.{layer}.attn.W_K"] = W_K
        out_sd[f"blocks.{layer}.attn.W_V"] = W_V
        # out_sd[f"blocks.{layer}.attn.b_Q"] = torch.zeros_like(W_Q)
        # out_sd[f"blocks.{layer}.attn.b_K"] = torch.zeros_like(W_K)
        # out_sd[f"blocks.{layer}.attn.b_V"] = torch.zeros_like(W_V)
        W_O = in_sd[f"{layer_key}.attn.c_proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        out_sd[f"blocks.{layer}.attn.W_O"] = W_O

        # MLP Weights
        out_sd[f"blocks.{layer}.mlp.W_in"] = in_sd[f"{layer_key}.mlp.c_fc.weight"].T
        # out_sd[f"blocks.{layer}.mlp.b_in"] = torch.zeros_like(in_sd[f"{layer_key}.mlp.c_fc.weight"][0])
        out_sd[f"blocks.{layer}.mlp.W_out"] = in_sd[f"{layer_key}.mlp.c_proj.weight"].T
        # out_sd[f"blocks.{layer}.mlp.b_out"] = torch.zeros_like(in_sd[f"{layer_key}.mlp.c_proj.weight"][0])

    return out_sd


if LOAD_AND_CONVERT_CHECKPOINT:
    synthetic_checkpoint = model_state
    for name, param in synthetic_checkpoint.items():
        if name.startswith("_orig_mod.transformer.h.0") or not name.startswith(
            "_orig_mod.transformer.h"
        ):
            print(name, param.shape)

    n_heads = 8
    n_layers = 16

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=512,
        d_head=64,
        n_heads=n_heads,
        d_mlp=2048,
        d_vocab=32,
        n_ctx=1023,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)
    model.to(device)

    model.load_and_process_state_dict(
        convert_to_transformer_lens_format(
            synthetic_checkpoint, n_layers=n_layers, n_heads=n_heads
        )
    )
    torch.save(model.state_dict(), f"{MODEL_DIR}tf_lens_{n_layers}.pth")

# An example input
sample_input = torch.tensor([[15, 6, 4, 27, 9, 0, 25, 10, 0, 7, 4, 19]]).to(device)
# sample_input = torch.tensor([[15, 6, 4, 27, 9]])
# The argmax of the output (ie the most likely next move from each position)
sample_output = torch.tensor([[6, 4, 27, 9, 0, 27, 10, 0, 7, 4, 19, 28]])
model_output = model(sample_input).argmax(dim=-1)
print(model_output)
print(sample_output == model_output)
assert torch.all(sample_output == model_output)

### END MODEL SETUP ###
