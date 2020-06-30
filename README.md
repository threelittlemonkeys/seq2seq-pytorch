# Sequence to Sequence Models in PyTorch

Minimal implementations of sequence to sequence models in PyTorch.

- RNN Encoder-Decoder (Cho et al 2014; Luong et al 2015; Gu et al 2016)
- Pointer Networks (Vinyals et al 2015)
- CNNs from "Convolutional Sequence to Sequence Learning" (Gehring et al 2017)
- The Transformer from "Attention Is All You Need" (Vaswani et all 2017)

## References

Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, Llion Jones. [Character-Level Language Modeling with Deeper Self-Attention.](https://arxiv.org/abs/1808.04444) arXiv:1808.04444.

Philip Arthur, Graham Neubig, Satoshi Nakamura. 2016. [Incorporating Discrete Translation Lexicons into Neural Machine Translation.](https://arxiv.org/abs/1606.02006) In EMNLP.

Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton. 2016. [Layer Normalization.](https://arxiv.org/abs/1607.06450) arXiv:1607.06450.

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. 2015. [Neural Machine Translation by Jointly Learning to Align and Translate.](https://arxiv.org/abs/1409.0473) arXiv:1409.0473.

James Bradbury, Stephen Merity, Caiming Xiong, Richard Socher. 2016. [Quasi-Recurrent Neural Networks.](https://arxiv.org/abs/1611.01576) arXiv:1611.01576.

Denny Britz, Anna Goldie, Minh-Thang Luong, Quoc Le. 2017. [Massive Exploration of Neural Machine Translation Architectures.](https://arxiv.org/abs/1703.03906) arXiv:1703.03906.

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio. 2014. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.](https://arxiv.org/abs/1406.1078) arXiv:1406.1078.

Andrew M. Dai, Quoc V. Le. [Semi-supervised Sequence Learning.](https://arxiv.org/abs/1511.01432) arXiv:1511.01432.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov. 2019. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.](https://arxiv.org/abs/1901.02860) In ACL.

Jiachen Du, Wenjie Li, Yulan He, Ruifeng Xu, Lidong Bing, Xuan Wang. 2018. [Variational Autoregressive Decoder for Neural Response Generation.](https://aclweb.org/anthology/D18-1354) In EMNLP.

Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin. 2017. [Convolutional Sequence to Sequence Learning.](https://arxiv.org/abs/1705.03122) arXiv:1705.03122.

Alex Graves. 2013. [Generating Sequences With Recurrent Neural Networks.](https://arxiv.org/abs/1308.0850) arXiv:1308.0850.

Jiatao Gu, Zhengdong Lu, Hang Li, Victor O.K. Li. 2016. [Incorporating Copying Mechanism in Sequence-to-Sequence Learning.](https://arxiv.org/abs/1603.06393) In ACL.

Jeremy Hylton. 1993. [The Complete Works of William Shakespeare.](http://shakespeare.mit.edu) http://shakespeare.mit.edu.

Łukasz Kaiser, Samy Bengio. 2018. [Discrete Autoencoders for Sequence Models.](https://arxiv.org/abs/1801.09797) arXiv:1801.09797.

Jing Li, Aixin Sun, Shafiq Joty. 2018. [SEGBOT: A Generic Neural Text Segmentation Model with Pointer Network.](https://www.ijcai.org/proceedings/2018/0579.pdf) In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence.

Jiwei Li. 2017. [Teaching Machines to Converse.](https://github.com/jiweil/Jiwei-Thesis/blob/master/thesis.pdf) Doctoral dissertation. Stanford University.

Junyang Lin, Xu Sun, Xuancheng Ren, Muyu Li, Qi Su. 2018. [Learning When to Concentrate or Divert Attention: Self-Adaptive Attention Temperature for Neural Machine Translation.](https://arxiv.org/abs/1808.07374) arXiv:1808.07374.

Minh-Thang Luong, Hieu Pham, Christopher D. Manning. 2015. [Effective Approaches to Attention-based Neural Machine Translation.](https://arxiv.org/abs/1508.04025) In EMNLP.

Xuezhe Ma, Zecong Hu, Jingzhou Liu, Nanyun Peng, Graham Neubig, Eduard Hovy. 2018. [Stack-Pointer Networks for Dependency Parsing.](https://aclweb.org/anthology/P18-1130). In ACL.

Hideya Mino, Masao Utiyama, Eiichiro Sumita, Takenobu Tokunaga. 2017. [Key-value Attention Mechanism for Neural Machine Translation.](http://aclweb.org/anthology/I17-2049) In Proceedings of the 8th International Joint Conference on Natural Language Processing.

Chan Young Park, Yulia Tsvetkov. [Learning to Generate Word- and Phrase-Embeddings for Efficient Phrase-Based Neural Machine Translation.](https://www.aclweb.org/anthology/D19-5626.pdf) In Proceedings of the 3rd Workshop on Neural Generation and Translation.

Ofir Press, Lior Wolf. 2016. [Using the Output Embedding to Improve Language Models.](https://arxiv.org/abs/1608.05859) arXiv:1608.05859.

Abigail See, Peter J. Liu, Christopher D. Manning. 2017. [Get To The Point: Summarization with Pointer-Generator Networks.](https://arxiv.org/abs/1704.04368) arXiv:1704.04368.

Xiaoyu Shen, Hui Su, Shuzi Niu, Vera Demberg. 2018. [Improving Variational Encoder-Decoders in Dialogue Generation.](https://arxiv.org/abs/1802.02032) In AAAI.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. 2017. [Attention Is All You Need.](https://arxiv.org/abs/1706.03762) In NIPS.

Oriol Vinyals, Meire Fortunato, Navdeep Jaitly. 2015. [Pointer Networks.](https://arxiv.org/abs/1506.03134) In NIPS.

Oriol Vinyals, Samy Bengio, Manjunath Kudlur. 2015. [Order Matters: Sequence to sequence for sets.](https://arxiv.org/abs/1511.06391) In ICLR.

Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, Jason Weston. 2019. [Neural Text Generation with Unlikelihood Training.](https://arxiv.org/abs/1908.04319) arXiv:1908.04319.

Sam Wiseman, Alexander M. Rush. [Sequence-to-Sequence Learning as Beam-Search Optimization.](https://arxiv.org/abs/1606.02960) arXiv:1606.02960.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, Jeffrey Dean. 2016. [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation.](https://arxiv.org/abs/1609.08144) arXiv:1609.08144.

Ziang Xie. 2018. [Neural Text Generation: A Practical Guide.](http://cs.stanford.edu/~zxie/textgen.pdf) http://cs.stanford.edu/~zxie/textgen.pdf.

Feifei Zhai, Saloni Potdar, Bing Xiang, Bowen Zhou. 2017. [Neural Models for Sequence Chunking.](https://arxiv.org/abs/1701.04027) In AAAI.

Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, Jason Weston. 2018. [Personalizing Dialogue Agents: I have a dog, do you have pets too?.](https://arxiv.org/abs/1801.07243) arXiv:1801.07243.
