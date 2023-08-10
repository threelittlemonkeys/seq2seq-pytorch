# RNN Encoder-Decoder in PyTorch

A minimal PyTorch implementation of RNN Encoder-Decoder for sequence to sequence learning.

Supported features:
- Mini-batch training with CUDA
- Lookup, CNNs, RNNs and/or self-attentive encoding in the embedding layer
- Input feeding (Luong et al 2015)
- Attention mechanism (Bahdanau et al 2014, Luong et al 2015)
- CopyNet, copying mechanism (Gu et al 2016)
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
