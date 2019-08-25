# RNN Encoder-Decoder in PyTorch

A minimal PyTorch implementation of the RNN Encoder-Decoder for sequence to sequence learning.

Supported features:
- Mini-batch training with CUDA
- Lookup, CNNs, RNNs and/or self-attentive encoding in the embedding layer
- Global and local attention (Luong et al 2015)
- Vectorized computation of alignment scores in the attention layer
- Input feeding (Luong et al 2015)
- Beam search decoding
- CopyNet (Gu et al 2016)

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
python train.py model vocab.src vocab.tgt training_data.csv num_epoch
```

To predict:
```
python predict.py model.epochN vocab.src vocab.tgt test_data
```
