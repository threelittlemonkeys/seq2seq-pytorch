# Pointer Networks in PyTorch

A minimal PyTorch implementation of Pointer Networks.

Supported features:
- Mini-batch training with CUDA
- Lookup, CNNs, RNNs and/or self-attentive encoding in the embedding layer
- Vectorized computation of alignment scores in the attention layer
- Beam search decoding

## Usage

Training data should be formatted as below:
```
source_sequence \t target_sequence
source_sequence \t target_sequence
...
```

To prepare data:
```
python prepare.py training_data
```

To train:
```
python3 train.py model vocab training_data.csv (validation_data) num_epoch
```

To predict:
```
python predict.py model.epochN vocab test_data
```

To evaluate:
```
python evaluate.py model.epochN vocab test_data
```
