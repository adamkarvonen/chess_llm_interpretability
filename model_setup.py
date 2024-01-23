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

import os

### BEGIN MODEL SETUP ###
# Our pytorch model is in the nanogpt format. For easy linear probing of the residual stream, we want to convert
# it to the transformer lens format. This is done in the following code block.
# This code was developed using Neel Nanda's othello_reference/Othello_GPT.ipynb as a reference.
# Once again, I just copy pasted the relevant cells into here for convenience. Sorry for the messiness.

torch.set_grad_enabled(False)

LOAD_AND_CONVERT_CHECKPOINT = True

device = "cpu"

MODEL_DIR = "models/"

model_name = "lichess_16layers_ckpt_no_optimizer.pt"

n_heads = 8
n_layers = 16  # NOTE: This is pretty janky. You can infer the number of layers from the model name, so use that to set this.

assert str(n_layers) in model_name

if not os.path.exists(f"{MODEL_DIR}{model_name}"):
    state_dict = utils.download_file_from_hf("adamkarvonen/chess_llms", model_name)
    model = torch.load(state_dict, map_location=device)
    torch.save(model, f"{MODEL_DIR}{model_name}")


checkpoint = torch.load(f"{MODEL_DIR}{model_name}", map_location=device)

# Print the keys of the checkpoint dictionary
print(checkpoint.keys())
model_state = checkpoint["model"]
# for key, value in model_state.items():
#     print(key, value.shape)


def convert_nanogpt_weights(
    old_state_dict, cfg: HookedTransformerConfig, bias: bool = False
):
    """For https://github.com/karpathy/nanoGPT
    There are two complications with converting nanogpt models:
    The first is that some state dicts have an unwanted prefix on keys that needs to be removed.
    The second is that the models can be saved with or without bias. By default, there
    is no bias. This function can handle both cases."""
    # Nanogpt models saved after torch.compile() have this unwanted prefix
    # This is a simple way to remove it
    unwanted_prefix = "_orig_mod."
    for k, v in list(old_state_dict.items()):
        if k.startswith(unwanted_prefix):
            old_state_dict[k[len(unwanted_prefix) :]] = old_state_dict.pop(k)

    new_state_dict = {}
    new_state_dict["pos_embed.W_pos"] = old_state_dict["transformer.wpe.weight"]
    new_state_dict["embed.W_E"] = old_state_dict["transformer.wte.weight"]

    new_state_dict["ln_final.w"] = old_state_dict["transformer.ln_f.weight"]
    new_state_dict["ln_final.b"] = torch.zeros_like(
        old_state_dict["transformer.ln_f.weight"]
    )
    new_state_dict["unembed.W_U"] = old_state_dict["lm_head.weight"].T

    if bias:
        new_state_dict["ln_final.b"] = old_state_dict["transformer.ln_f.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"transformer.h.{layer}"

        new_state_dict[f"blocks.{layer}.ln1.w"] = old_state_dict[
            f"{layer_key}.ln_1.weight"
        ]
        # A bias of zeros is required for folding layer norm
        new_state_dict[f"blocks.{layer}.ln1.b"] = torch.zeros_like(
            old_state_dict[f"{layer_key}.ln_1.weight"]
        )
        new_state_dict[f"blocks.{layer}.ln2.w"] = old_state_dict[
            f"{layer_key}.ln_2.weight"
        ]
        new_state_dict[f"blocks.{layer}.ln2.b"] = torch.zeros_like(
            old_state_dict[f"{layer_key}.ln_2.weight"]
        )

        W = old_state_dict[f"{layer_key}.attn.c_attn.weight"]
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=0)
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        new_state_dict[f"blocks.{layer}.attn.W_Q"] = W_Q
        new_state_dict[f"blocks.{layer}.attn.W_K"] = W_K
        new_state_dict[f"blocks.{layer}.attn.W_V"] = W_V

        W_O = old_state_dict[f"{layer_key}.attn.c_proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        new_state_dict[f"blocks.{layer}.attn.W_O"] = W_O

        new_state_dict[f"blocks.{layer}.mlp.W_in"] = old_state_dict[
            f"{layer_key}.mlp.c_fc.weight"
        ].T
        new_state_dict[f"blocks.{layer}.mlp.W_out"] = old_state_dict[
            f"{layer_key}.mlp.c_proj.weight"
        ].T

        if bias:
            new_state_dict[f"blocks.{layer}.ln1.b"] = old_state_dict[
                f"{layer_key}.ln_1.bias"
            ]
            new_state_dict[f"blocks.{layer}.ln2.b"] = old_state_dict[
                f"{layer_key}.ln_2.bias"
            ]
            new_state_dict[f"blocks.{layer}.mlp.b_in"] = old_state_dict[
                f"{layer_key}.mlp.c_fc.bias"
            ]
            new_state_dict[f"blocks.{layer}.mlp.b_out"] = old_state_dict[
                f"{layer_key}.mlp.c_proj.bias"
            ]

            B = old_state_dict[f"{layer_key}.attn.c_attn.bias"]
            B_Q, B_K, B_V = torch.tensor_split(B, 3, dim=0)
            B_Q = einops.rearrange(B_Q, "(i h)->i h", i=cfg.n_heads)
            B_K = einops.rearrange(B_K, "(i h)->i h", i=cfg.n_heads)
            B_V = einops.rearrange(B_V, "(i h)->i h", i=cfg.n_heads)
            new_state_dict[f"blocks.{layer}.attn.b_Q"] = B_Q
            new_state_dict[f"blocks.{layer}.attn.b_K"] = B_K
            new_state_dict[f"blocks.{layer}.attn.b_V"] = B_V
            new_state_dict[f"blocks.{layer}.attn.b_O"] = old_state_dict[
                f"{layer_key}.attn.c_proj.bias"
            ]

    return new_state_dict


if LOAD_AND_CONVERT_CHECKPOINT:
    synthetic_checkpoint = model_state
    for name, param in synthetic_checkpoint.items():
        if name.startswith("_orig_mod.transformer.h.0") or not name.startswith(
            "_orig_mod.transformer.h"
        ):
            print(name, param.shape)

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
        convert_nanogpt_weights(synthetic_checkpoint, cfg)
    )
    recorded_model_name = model_name.split(".")[0]
    torch.save(model.state_dict(), f"{MODEL_DIR}tf_lens_{recorded_model_name}.pth")

# An example input
sample_input = torch.tensor([[15, 6, 4, 27, 9, 0, 25, 10, 0, 7, 4, 19]]).to(device)
# sample_input = torch.tensor([[15, 6, 4, 27, 9]])
# The argmax of the output (ie the most likely next move from each position)
sample_output = torch.tensor([[6, 4, 27, 9, 0, 27, 10, 0, 7, 4, 19, 28]])
model_output = model(sample_input).argmax(dim=-1)
print(model_output)
print(sample_output == model_output)

# For this particular sample_input, any model with decent chess skill should output sample_output.
# So, this assert will definitely fail for a randomly initialized model, and may fail for models with low skill.
# But, I've never seen that happen, so I'm keeping it simple for now. For a more robust test, use the nanogpt_to_transformer_lens.ipynb notebook.
# This notebook actually runs the sample input through the original nanogpt model, and then through the converted transformer lens model.
assert torch.all(sample_output == model_output)

### END MODEL SETUP ###
