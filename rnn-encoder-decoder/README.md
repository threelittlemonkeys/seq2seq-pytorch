# The RNN Encoder-Decoder in PyTorch

A PyTorch implementation of the RNN Encoder-Decoder for sequence to sequence learning, adapted from [the PyTorch tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

Supported features:
- Mini-batch training with CUDA
- Global and local attention (Luong et al 2015)
- Input feeding (Luong et al 2015)
- Vectorized computation of alignment scores in the attention layer

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
