# chess_llm_interpretability
This evaluates llms trained on PGN format chess games and evaluates board understand, similar to the Othello World paper.

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

Much of my linear probing was developed using Neel Nanda's linear probing code as a reference. They can be found at:

https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Othello_GPT.ipynb
https://colab.research.google.com/github/likenneth/othello_world/blob/master/Othello_GPT_Circuits.ipynb
https://www.neelnanda.io/mechanistic-interpretability/othello
https://github.com/likenneth/othello_world/tree/master/mechanistic_interpretability