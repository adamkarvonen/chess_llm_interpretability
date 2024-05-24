# chess_llm_interpretability
This evaluates LLMs trained on PGN format chess games through the use of linear probes. We can check the LLMs internal understanding of board state and ability to estimate the skill level of the players involved. We can also perform interventions on the model's internal board state by deleting pieces from its internal world model.

This repo can train, evaluate, and visualize linear probes on LLMs that have been trained to play chess with PGN strings. For example, we can visualize where the model "thinks" the white pawns are. On the left, we have the actual white pawn location. In the middle, we clip the probe outputs to turn the heatmap into a more binary visualization. On the right, we have the full gradient of model beliefs.

![](/images/pawn_probe.png)

I trained linear probes on both the model's ability to compute board state and estimate player ELO as it's predicting the next character. Here we can see a per layer graph of board state and elo classification accuracy across a range of LLMs.

![](/images/probe_acc_markers_graph.png)

For more information, refer to this [post](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html).

# Setup

Create a Python environment with Python 3.10 or 3.11 (I'm using 3.11).
```
pip install -r requirements.txt
python model_setup.py
```

Then click "Run All" on `lichess_data_filtering.ipynb` (I'm filtering data in a notebook instead of a script because I use a series of graphs to illustrate what the data filtering is doing).
To visualise probe outputs or better understand my work, check out `probe_output_visualization.ipynb`. It has commentary and many print statements to walk you through using a single probe and performing a single intervention.

The `train_test_chess.py` script can be used to either train new linear probes or test a saved probe on the test set.

Command line arguments:

--mode: Specifies `train`  or `test`. Optional, defaults to `train`.

--probe: Determines the type of probe to be used. `piece` probes for the piece type on each square, `skill` probes the skill level of the White player. Optional, defaults to `piece`.


Examples:

Train piece board state probes:
`python train_test_chess.py`

Test skill probe:
`python train_test_chess.py --mode test --probe skill`

See all options: `python train_test_chess.py -h`

To add new functions, refer to `utils/custom_functions_guide.md`.

All experiments in this repo can be done with less than 1 GB of VRAM. Training probes on the 8 layer model takes about 10 minutes on my RTX 3050.

# OthelloGPT

This repo can also be used for training linear probes on OthelloGPT. Refer to `utils/othello_data_filtering.ipynb`.

# Interventions

To perform board state interventions on one layer, run `python board_state_interventions.py`. It will record JSON results in `intervention_logs/`. To get better results, train a set of 8 (one per layer) board state probes using `train_test_chess.py` and rerun.

To perform skill interventions, you can train a set of 8 skill probes using `train_test_chess.py` or generate a set of 8 contrastive activations using `caa.py`. Note that contrastive activations tend to work a little better. If you want to use probe derived interventions, use this script to create activation files from the probes: `utils/create_skill_intervention_from_skill_probe.ipynb`.

Then, follow these directions to use them to perform skill interventions: https://github.com/adamkarvonen/chess_gpt_eval/tree/master/nanogpt

# Shape Annotations

I've been using this tip from Noam Shazeer:

Dimension key (from https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):

M = modes

l  = seq length before indexing

L  = seq length after indexing

B = batch_size

R = rows (or cols)

C = classes for one hot encoding

D = d_model of the GPT (512)

For example

```
probe_out_MBLRRC = einsum(
    "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
    resid_post_BLD,
    linear_probe_MDRRC,
)
```

# Useful links

All code, models, and datasets are open source.

To play the nanoGPT model against Stockfish, please visit: https://github.com/adamkarvonen/chess_gpt_eval/tree/master/nanogpt

To train a Chess-GPT from scratch, please visit: https://github.com/adamkarvonen/nanoGPT

All pretrained models are available here: https://huggingface.co/adamkarvonen/chess_llms

All datasets are available here: https://huggingface.co/datasets/adamkarvonen/chess_games

Wandb training loss curves and model configs can be viewed here: https://api.wandb.ai/links/adam-karvonen/u783xspb

# Testing

To run the end to end test suite, run `pytest -s` from the root directory. This will first train and test probes end to end on the 8 layer model, including comparing expected accuracy to actual accuracy within some tolerance. Then it will test out board state interventions and caa creation. It takes around 14 minutes. The `-s` flag is so you can see the training updates and gauge progress.

# References

Much of my linear probing was developed using Neel Nanda's linear probing code as a reference. Here are the main references I used:

https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Othello_GPT.ipynb
https://colab.research.google.com/github/likenneth/othello_world/blob/master/Othello_GPT_Circuits.ipynb
https://www.neelnanda.io/mechanistic-interpretability/othello
https://github.com/likenneth/othello_world/tree/master/mechanistic_interpretability