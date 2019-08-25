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
python train.py model vocab.src vocab.tgt training_data.csv num_epoch
```

To predict:
```
python predict.py model.epochN vocab.src vocab.tgt test_data

## References

Xuezhe Ma, Zecong Hu, Jingzhou Liu, Nanyun Peng, Graham Neubig, Eduard Hovy. 2018. [Stack-Pointer Networks for Dependency Parsing.](https://aclweb.org/anthology/P18-1130). In ACL.

Abigail See, Peter J. Liu, Christopher D. Manning. 2017. [Get To The Point: Summarization with Pointer-Generator Networks.](https://arxiv.org/abs/1704.04368) arXiv:1704.04368.

Oriol Vinyals, Meire Fortunato, Navdeep Jaitly. 2015. [Pointer Networks.](https://arxiv.org/abs/1506.03134) In NIPS.
