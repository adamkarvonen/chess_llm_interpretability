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

Then click "Run All" on `lichess_data_filtering.ipynb`.
To visualise probe outputs, check out `probe_output_visualization.ipynb`.

To train a linear probe or test a saved probe on the test set, set these two variables at the bottom of `train_test_chess.py`:
RUN_TEST_SET = True
USE_PIECE_BOARD_STATE = True

Then run `python train_test_chess.py`.

All experiments in this repo can be done with less than 1 GB of VRAM. Training probes on the 8 layer model takes about 10 minutes on my RTX 3050.

# Interventions

To perform board state interventions, first train a set of 8 (one per layer) board state probes using `train_test_chess.py`. Then run `python board_state_interventions.py`. It will record JSON results in `intervention_logs/`.

To perform skill interventions, you can train a set of 8 skill probes using `train_test_chess.py` or generate a set of 8 contrastive activations using `caa.py`. Note that contrastive activations tend to work a little better. If you want to use probe derived interventions, use this script to create activation files from the probes: `utils/create_skill_intervention_from_skill_probe.ipynb`.

Then, follow these directions to use them to perform skill interventions: https://github.com/adamkarvonen/chess_gpt_eval/tree/master/nanogpt

# Useful links

All code, models, and datasets are open source.

To play the nanoGPT model against Stockfish, please visit: https://github.com/adamkarvonen/chess_gpt_eval/tree/master/nanogpt

To train a Chess-GPT from scratch, please visit: https://github.com/adamkarvonen/nanoGPT

All pretrained models are available here: https://huggingface.co/adamkarvonen/chess_llms

All datasets are available here: https://huggingface.co/datasets/adamkarvonen/chess_games

Wandb training loss curves and model configs can be viewed here: https://api.wandb.ai/links/adam-karvonen/u783xspb

# Testing

To run the end to end test suite, run `pytest -s` from the root directory. This will train and test probes end to end on the 8 layer model, including comparing expected accuracy to actual accuracy within some tolerance. It takes around 12 minutes. The `-s` flag is so you can see the training updates and gauge progress.

# References

Much of my linear probing was developed using Neel Nanda's linear probing code as a reference. Here are the main references I used:

https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Othello_GPT.ipynb
https://colab.research.google.com/github/likenneth/othello_world/blob/master/Othello_GPT_Circuits.ipynb
https://www.neelnanda.io/mechanistic-interpretability/othello
https://github.com/likenneth/othello_world/tree/master/mechanistic_interpretability