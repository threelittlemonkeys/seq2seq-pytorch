# Transformer in PyTorch

A minimal PyTorch implementation of Transformer for sequence to sequence learning.

Supported features:
- Mini-batch training with CUDA
- Pre layer normalization (Wang et al 2019)
- Beam search decoding
- Attention visualization

## Usage

Training data should be formatted as below:
```
source_sequence \t target_sequence
source_sequence \t target_sequence
...
```

To prepare data:
```
python3 prepare.py training_data
```

To train:
```
python3 train.py model vocab.src vocab.tgt training_data.csv num_epoch
```

To predict:
```
python3 predict.py model.epochN vocab.src vocab.tgt test_data
```
