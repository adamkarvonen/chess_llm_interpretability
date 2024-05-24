If you wish to add a new Othello or Chess board to state function, there are three steps. I will use `board_to_pin_state` and `games_batch_to_state_stack_mine_yours_BLRRC` as my examples.

First, write a function that converts a chess board or othello board to a pytorch tensor. Refer to `chess_utils.py/board_to_pin_state()` and `othello_utils.py/games_batch_to_state_stack_mine_yours_BLRRC()` for how to do this.

Next, add a `Config` for this function. To do this, refer to `chess_utils.py/othello_config` and `chess_utils.py/pin_config`. This config object is primarily used to indicate the expected size of the one hot output tensor, the name for the linear probe, if the function is used with othello, and if the board to state function should get additional data such as skill. I'm not very happy about this implementation, and it really should get refactored so board to state functions return one hot tensors instead.

Currently, board to state functions return a row by column tensor. For board state, this would be 8x8. In Othello, if an element was -1, then it's white on that square. If it's 1, then it's black on that square. So, then tensor is 8x8, and all elements are -1, 0, or 1.

I had done this so I could easily plot and visualize board states. This was a mistake. Everything should return one hot tensors, and it should convert one hot to standard tensors for plotting.

Here's `othello_config`:

```
othello_config = Config(
    min_val=-1,
    max_val=1,
    custom_board_state_function=othello_utils.games_batch_to_state_stack_mine_yours_BLRRC,
    linear_probe_name="othello_mine_yours_probe",
    othello=True,
)
```

The minimum and maximum values present in the board state are -1 and 1, so that's min val and max val. After one hot encoding, it will be shape (8, 8, 3). The 3 is because any square can be -1, 0, or 1. Here's `pin_config`:

```
pin_config = Config(
    min_val=0,
    max_val=1,
    custom_board_state_function=board_to_pin_state,
    num_rows=1,
    num_cols=1,
    linear_probe_name="chess_pin_probe",
)
```

Because pin config is a 1x1 binary variable, its shape will be (1, 1, 1). It still has rows and columns so the shapes work with einops.

Finally, in `train_test_chess.py`, add `config = chess_utils.my_config` before this line: `input_dataframe_file = f"{DATA_DIR}{dataset_prefix}{split}.csv"`. 

This is all you need to train a linear probe. To test a linear probe, add your config to `chess_utils.py/all_configs` in `chess_utils.py/find_config_by_name()`.

If I every get around to it, this could be significantly cleaned up. The codebase started evolving and I never got around to doing a refactor to clean it up. It's not terrible as it's just three steps. For now, it is what it is.