# Sorting Numbers

This is a simple tutorial for sorting numbers with Pointer Networks.

1. Training and validation data should be formatted as below, numbers and their sorted indices separated by `\t`:

```
1 2 3 \t 0 1 2
1 3 2 \t 0 2 1
2 1 3 \t 1 0 2
2 3 1 \t 1 2 0
3 1 2 \t 2 0 1
3 2 1 \t 2 1 0
...
```

2. Modify the type and dimension of the word embedding layer in `parameters.py`, for exmaple:

```
UNIT = "word"
EMBED = {"rnn": 100}
```

4. Run `prepare.py` to make CSV and index files.

```
python3 ../prepare.py train
```

5. Train your model. You can modify the hyperparameters in `parameters.py`.

```
python3 ../train.py model train.char_to_idx train.word_to_idx train.csv valid N
```

6. Predict and evaluate your model.

```
python3 predict.py model.epochN train.char_to_idx train.word_to_idx test
python3 evaluate.py model.epochN train.char_to_idx train.word_to_idx test
```
