# chess_llm_interpretability
This evaluates LLMs trained on PGN format chess games and evaluates board understanding, similar to the Othello World paper.

This repo can train, evaluate, and visualize linear probes on LLMs that have been trained to play chess with PGN strings. For example, we can visualize where the model "thinks" the white pawns are. On the left, we have the actual white pawn location. In the middle, we clip the probe outputs to turn the heatmap into a more binary visualization. On the right, we have the full gradient of model beliefs, and we can see it's extremely confident that no white pawns are on either side's back rank.

![](/images/pawn_probe.png)

I trained linear probes on the model's ability to estimate player ELO as it's predicting the next character. Here we can see a graph of ELO classification accuracy per layer of the LLM.

![](/images/accuracy_per_layer_elo.png)

For more information, refer to this [post](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html).

# Setup

Create a Python environment with Python 3.10.
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

# References

Much of my linear probing was developed using Neel Nanda's linear probing code as a reference. Here are the main references I used:

https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Othello_GPT.ipynb
https://colab.research.google.com/github/likenneth/othello_world/blob/master/Othello_GPT_Circuits.ipynb
https://www.neelnanda.io/mechanistic-interpretability/othello
https://github.com/likenneth/othello_world/tree/master/mechanistic_interpretability