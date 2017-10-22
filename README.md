# Awesome Chainer [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

## What is Chainer?

Chainer is a flexible framework for neural networks. One major goal is flexibility, so it must enable us to write complex architectures simply and intuitively.

More info  [here](http://chainer.org/)

ref [Chainer.wiki](https://github.com/pfnet/chainer/wiki) - Chainer Wiki

## Table of Contents

<!-- MarkdownTOC depth=4 -->
- [Tutorials](#github-tutorials)
- [Models/Projects](#github-projects)
- [Examples](#github-examples)
- [Libraries](#libraries)
- [Videos](#video)
- [Papers](#papers)
- [Blog posts](#blogs)
- [Community](#community)
- [Books](#books)

<!-- /MarkdownTOC -->


<a name="github-tutorials" />

## Tutorials

* [Chainer Tutorial Offical document]http://docs.chainer.org/en/latest/tutorial/index.html) - Chainer Tutorial Offical document

## Hands-on

*	[hido/chainer-handson](https://github.com/hido/chainer-handson/blob/master/chainer.ipynb) - Chainer Hands-on
*	[iwiwi/chainer-handson](https://github.com/iwiwi/chainer-handson/blob/master/chainer-ja.ipynb) - Chainer Hands-on(JP CPU only)



<a name="github-projects" />

## Models/Projects

### Preferred Networks official

* [ChainerRL](https://github.com/pfnet/chainerrl) - ChainerRL is a deep reinforcement learning library built on top of Chainer.
* [ChainerCV](https://github.com/pfnet/chainercv) -  Versatile set of tools for Deep Learning based Computer Vision
* [Paint Chainer](https://github.com/pfnet/PaintsChainer) - Paints Chainer is line drawing colorizer using chainer.

<a name="#github-examples" />

# Chainer External Examples

## Preferred NetWorks Research
* [LSGAN](https://github.com/pfnet-research/chainer-LSGAN) - Least Squares Generative Adversarial Network implemented in Chainer
* [Chainer-gogh](https://github.com/pfnet-research/chainer-gogh) - Implementation of "A neural algorithm of Artistic style"
* [Chainer Graph CNN](https://github.com/pfnet-research/chainer-graph-cnn) - Chainer implementation of 'Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering' (https://arxiv.org/abs/1606.09375)
* [chainer-segnet](https://github.com/pfnet-research/chainer-segnet) - SegNet implementation & experiments in Chainer
* [chainer-pix2pix](https://github.com/pfnet-research/chainer-pix2pix) - chainer implementation of pix2pix


## Examples
| Method | Codes |
|:--------|:------|
| Chainer Hands-on | [hido/chainer-handson](https://github.com/hido/chainer-handson/blob/master/chainer.ipynb) |
| Recurrent neural network (RNN) | [yusuketomoto/chainer-char-rnn](https://github.com/yusuketomoto/chainer-char-rnn) |
| Fast R-CNN | [mitmul/chainer-fast-rcnn](https://github.com/mitmul/chainer-fast-rcnn) |
| Fast R-CNN | [apple2373/chainer-simple-fast-rnn](https://github.com/apple2373/chainer-simple-fast-rnn) |
| Faster R-CNN | [mitmul/chainer-faster-rcnn](https://github.com/mitmul/chainer-faster-rcnn) |
| QRNN | [jekbradbury/qrnn.py](http://metamind.io/research/new-neural-network-building-block-allows-faster-and-more-accurate-text-understanding/) |
| Siamese Network | [mitmul/chainer-siamese](https://github.com/mitmul/chainer-siamese) |
| Support Vector Machine (SVM) | [mitmul/chainer-svm](https://github.com/mitmul/chainer-svm) |
| Group Equivariant Convolutional Neural Networks | [tscohen/GrouPy](https://github.com/tscohen/GrouPy) |


### NLP
| Method | Codes |
|:--------|:------|
| Machine Translation, word segmentation, and language model | [odashi/chainer_examples](https://github.com/odashi/chainer_examples) |
| RNN Language Model | [odashi/chainer_rnnlm.py](https://gist.github.com/odashi/0d6e259abcc14f2d2d28) |
| Neural Encoder-Decoder Machine Translation | [odashi/chainer_encoder_decoder.py](https://gist.github.com/odashi/8d21f8fc23c075cd3042) |
| LSTM variants | [prajdabre/chainer_examples](https://github.com/prajdabre/chainer_examples/blob/master/chainer-1.5/LSTMVariants.py) |

### Computer Vision

| Method | Codes |
|:--------|:------|
| Cifar10 | [mitmul/chainer-cifar10](https://github.com/mitmul/chainer-cifar10) |
| Deep pose | [mitmul/DeepPose](https://github.com/mitmul/deeppose) |
| Image super-resolution | [mrkn/chainer-srcnn](https://github.com/mrkn/chainer-srcnn) |
| Image super-resolution | [Hi-king/chainer_superresolution](https://github.com/Hi-king/chainer_superresolution) |
| Overfeat | [darashi/chainer-example-overfeat-classify](https://github.com/darashi/chainer-example-overfeat-classify) |
| Variational autoencoder (VAE) | [RyotaKatoh/chainer-Variational-AutoEncoder](https://github.com/RyotaKatoh/chainer-Variational-AutoEncoder) |
| VGG | [mitmul/chainer-imagenet-vgg](https://github.com/mitmul/chainer-imagenet-vgg) |
| ResNet | [yasunorikudo/chainer-ResNet](https://github.com/yasunorikudo/chainer-ResNet) |
| DenseNet | [yasunorikudo/chainer-DenseNet](https://github.com/yasunorikudo/chainer-DenseNet) |
| ResDrop | [yasunorikudo/chainer-ResDrop](https://github.com/yasunorikudo/chainer-ResDrop) |
| Convolution Filter Visualization | [mitmul/chainer-conv-vis](https://github.com/mitmul/chainer-conv-vis) |
| Perceptual Losses for Real-Time Style Transfer and Super-Resolution | [yusuketomoto/chainer-fast-neuralstyle](https://github.com/yusuketomoto/chainer-fast-neuralstyle) |
| illustration2vec | [rezoo/illustration2vec](https://github.com/rezoo/illustration2vec) |
| StyleNet (A Neural Algorithm of Artistic Style) | [apple2373/chainer_stylenet](https://github.com/apple2373/chainer_stylenet) |
| StyleNet (A Neural Algorithm of Artistic Style) | [mattya/chainer-gogh](https://github.com/mattya/chainer-gogh) |
| Show and Tell | [apple2373/chainer_caption_generation](https://github.com/apple2373/chainer_caption_generation) |
| Deep Predictive Coding Networks | [chainer-prednet](https://github.com/kunimasa-kawasaki/chainer-prednet) |
| BinaryNet | [hillbig/binary_net](https://github.com/hillbig/binary_net) |
| Variational Auto-Encoder (VAE), Generative Adversarial Nets (GAN), VAE-GAN | [stitchfix/fauxtograph](https://github.com/stitchfix/fauxtograph) |
| Generative Adversarial Nets (GAN) | [rezoo/data.py](https://gist.github.com/rezoo/4e005611aaa4dad26697) |
| Deep Convolutional Generative Adversarial Network (DCGAN) | [mattya/chainer-DCGAN](https://github.com/mattya/chainer-DCGAN) |
| Fluid simulation | [mattya/chainer-fluid](https://github.com/mattya/chainer-fluid) |
| Convolutional Autoencoder | [ktnyt/chainer_ca.py](https://gist.github.com/ktnyt/58e015dd9ff33049da5a) |
| Stacked Denoising Autoencoder | [tochikuji/chainer-libDNN](https://github.com/tochikuji/chainer-libDNN/blob/master/examples/mnist/SdA.py) |
| Recurrent Attention Model | [masaki-y/ram](https://github.com/masaki-y/ram) |
| Fully Convolutional Networks | [wkentaro/fcn](https://github.com/wkentaro/fcn) |
| Generative Adversarial Networks with Denoising Feature Matching | [hvy/chainer-gan-denoising-feature-matching](https://github.com/hvy/chainer-gan-denoising-feature-matching) |
| Visualizing and Understanding Convolutional Networks | [hvy/chainer-visualization](https://github.com/hvy/chainer-visualization) |
| Chainer GAN Trainer | [hvy/chainer-gan-trainer](https://github.com/hvy/chainer-gan-trainer) |
| Deep Feature Interpolation for Image Content Changes | [dsanno/chainer-dfi](https://github.com/dsanno/chainer-dfi) |
| SegNet | [mitmul/chainer-segnet](https://github.com/mitmul/chainer-segnet) |
| WGAN | [musyoku/wasserstein-gan](https://github.com/musyoku/wasserstein-gan) |
| IMSAT | [weihua916/imsat](https://github.com/weihua916/imsat) |
| SSD | [Hakuyume/chainer-ssd](https://github.com/Hakuyume/chainer-ssd) |
| YOLOv2 | [leetenki/YOLOv2](https://github.com/leetenki/YOLOv2) |
| YOLOtiny_v2 | [leetenki/YOLOtiny_v2](https://github.com/leetenki/YOLOtiny_v2_chainer) |
| Deformable-conv |[deformable-conv](https://github.com/yuyu2172/deformable-conv) |
| chainercv | [chainercv](https://github.com/pfnet/chainercv) |


### Reinforcement Learning
| Method | Codes |
|:--------|:------|
| Deep Q-Network (DQN) | [ugo-nama-kun/DQN-chainer](https://github.com/ugo-nama-kun/DQN-chainer) |

### musyoku
| Method | Codes |
|:--------|:------|
| LSGAN | [musyoku/LSGAN](https://github.com/musyoku/LSGAN) |
| BEGAN | [musyoku/began](https://github.com/musyoku/began) |
| adversarial-autoencoder |  [adversarial-autoencoder](https://github.com/musyoku/adversarial-autoencoder) |
| chainer-dfi | [dsanno/chainer-dfi](https://github.com/dsanno/chainer-dfi) |
| chainer-VAE | [chainer-VAE](https://github.com/crcrpar/chainer-VAE) |
| wavenet | [wavenet](https://github.com/musyoku/wavenet) |
| chainer-sequential | [chainer-sequential](https://github.com/musyoku/chainer-sequential) |
| IMSAT | [musyoku/IMSAT](https://github.com/musyoku/IMSAT) |
| unrolled-gan| [musyoku/unrolled-gan](https://github.com/musyoku/unrolled-gan) |
| improved-gan | [musyoku/improved-gan](https://github.com/musyoku/improved-gan)|
| mnist-oneshot | [musyoku/mnist-oneshot](https://github.com/musyoku/mnist-oneshot) |
| adgm | [musyoku/adgm](https://github.com/musyoku/adgm) |
| VAT | [musyoku/vat](https://github.com/musyoku/vat) |
| DDGM | [musyoku/ddgm](https://github.com/musyoku/ddgm) |
| recurrent-batch-normalization | [musyoku/recurrent-batch-normalization](https://github.com/musyoku/recurrent-batch-normalization) |
| weight-normalization | [musyoku/weight-normalization](https://github.com/musyoku/weight-normalization) |
| minibatch_discrimination | [musyoku/minibatch_discrimination](https://github.com/musyoku/minibatch_discrimination) | 
| VAE | [musyoku/variational-autoencoder](https://github.com/musyoku/variational-autoencoder) |


## Blog posts

- [Introduction to Chainer: Neural Networks in Python](http://multithreaded.stitchfix.com/blog/2015/12/09/intro-to-chainer/)
- [The DIY Guide to Chainer](https://github.com/jxieeducation/DIY-Data-Science/blob/master/frameworks/chainer.md)
- [CHAINER CHARACTER EMBEDDINGS](http://dirko.github.io/Chainer-character-embeddings/)
- [A Fontastic Voyage: Generative Fonts with Adversarial Networks](http://multithreaded.stitchfix.com/blog/2016/02/02/a-fontastic-voyage/)


## Tools and extensions

* [Deel; A High level deep neural network description language](https://github.com/uei/deel)
* [DEEPstation](https://libraries.io/github/uei/deepstation)
* [chainer_imagenet_tools](https://github.com/shi3z/chainer_imagenet_tools)
* [scikit-chainer](https://github.com/lucidfrontier45/scikit-chainer)
* [chainer-libDNN](https://github.com/tochikuji/chainer-libDNN)



<a name="video" />

## Videos

### Only Japanese

* [Chainer の Trainer 解説とNStepLSTM について](https://www.youtube.com/watch?v=ok_bvPKAEaM) Published on Mar 15, 2017
* [Chainer Meetup #04](https://www.youtube.com/watch?v=Fq5ZQ1ccG38&t=6837s) Published on Feb 23, 2017
* [1014：深層学習フレームワークChainerの導入と化合物活性予測への応用](https://www.youtube.com/watch?v=lM76gLQag4I&t=1211s) Published on Dec 2, 2015



<a name="papers" />

## Papers

| Conference | Paper title | Codes | comments |
|:-----------|:------------|:------|:---------|
| arXiv only |  [GP-GAN: Towards Realistic High-Resolution Image Blending](https://arxiv.org/abs/1703.07195) | [wuhuikai/GP-GAN](https://github.com/wuhuikai/GP-GAN) | |
| arXiv only |  [Temporal Generative Adversarial Nets](https://arxiv.org/abs/1611.06624) | | |
| arXiv only |  [Reasoning with Memory Augmented Neural Networks for Language Comprehension](https://arxiv.org/abs/1610.06454) | | |
| arXiv only |  [PMI Matrix Approximations with Applications to Neural Language Modeling](https://arxiv.org/abs/1609.01235) | | |
| arXiv only |  [Neural Tree Indexers for Text Understanding](https://arxiv.org/abs/1607.04492) | [NTI](https://bitbucket.org/tsendeemts/nti/src) | |
| arXiv only |  [Neural Semantic Encoders](https://arxiv.org/abs/1607.04315) | | |
| arXiv only |  [Networked Intelligence: Towards Autonomous Cyber Physical Systems](https://arxiv.org/abs/1606.04087) | | |
| arXiv only |  [Modeling the dynamics of human brain activity with recurrent neural networks](https://arxiv.org/abs/1606.03071) | | |
| arXiv only |  [A Deep-Learning Approach for Operation of an Automated Realtime Flare Forecast](https://arxiv.org/abs/1606.01587) | | |
| arXiv only |  [Convolutional Neural Networks using Logarithmic Data Representation](https://arxiv.org/abs/1603.01025)| | |
| CoNLL 2016 |  [context2vec: Learning Generic Context Embedding with Bidirectional LSTM](http://u.cs.biu.ac.il/%7Emelamuo/publications/context2vec_conll16.pdf) | | |
|  CoNLL 2016  | [Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec](https://arxiv.org/abs/1605.02019) | | |
|  ECCV 2016 Workshop  | [Deep Impression: Audiovisual Deep Residual Networks for Multimodal Apparent Personality Trait Recognition](https://arxiv.org/abs/1609.05119) | | “3rd place in Looking at People ECCV Challenge” |
|  ECCV 2016 Workshop  | [Learning Joint Representations of Videos and Sentences with Web Image Search](https://arxiv.org/abs/1608.02367) | | |
| EMNLP 2016 | [Incorporating Discrete Translation Lexicons into Neural Machine Translation](https://arxiv.org/abs/1606.02006) | | |
| EMNLP 2016 | [Controlling Output Length in Neural Encoder-Decoders](https://arxiv.org/abs/1609.09552) | | |
| EMNLP 2016 | [Insertion Position Selection Model for Flexible Non-Terminals in Dependency Tree-to-Tree Machine Translation](http://www.aclweb.org/anthology/D16-1247) | | |
|  ICLR 2016  | [Learning Representations Using Complex-Valued Nets](https://arxiv.org/abs/1511.06351) | | |
|  ICLR 2017 under review  | [Dynamic Coattention Networks For Qustion Answering](https://arxiv.org/abs/1611.01604) | | |
|  ICLR 2017 under review  | [SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and < 0.5MB Model Size](https://arxiv.org/abs/1602.07360) | | |
|  ICLR 2017 under review  | [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576) | | |
|  ICLR 2017  | [Steerable CNNs](https://arxiv.org/abs/1612.08498) | | Chainer is not referred in the paper, but the authors kindly informed us.|
|  NIPS 2016 Workshop  | [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://arxiv.org/abs/1606.00709) | | |
|  OPT 2016  | [QSGD: Randomized Quantization for Communication-Optimal Stochastic Gradient Descent](https://arxiv.org/abs/1610.02132) | | |
|  PETRA 2016  | [Evaluation of Deep Learning based Pose Estimation for Sign Language Recognition](https://arxiv.org/abs/1602.09065) | | |
|  PASJ 2016  | [Machine-learning Selection of Optical Transients in Subaru/Hyper Suprime-Cam Survey](https://arxiv.org/abs/1609.03249) | | PASJ: Publications of the Astronomical Society of Japan |
|  Space Weather 2016  | [A Deep-Learning Approach for Operation of an Automated Realtime Flare Forecast](https://arxiv.org/abs/1606.01587) | | |
| NAACL 2016 | [Dynamic Entity Representation with Max-pooling Improves Machine Reading](http://aclweb.org/anthology/N/N16/N16-1099.pdf) | | |
| SemEval 2016 | [Feature-based Model versus Convolutional Neural Network for Stance Detection](http://aclweb.org/anthology/S/S16/S16-1065.pdf) | | |
| ACL 2016 | [Cross-Lingual Image Caption Generation](https://www.aclweb.org/anthology/P/P16/P16-1168.pdf) | | |
| ACL 2016 | [Composing Distributed Representations of Relational Patterns](http://www.aclweb.org/anthology/P16-1215) | | |
| ACL 2016 | [Generating Natural Language Descriptions for Semantic Representations of Human Brain Activity](https://www.aclweb.org/anthology/P/P16/P16-3004.pdf) | | |
| WMT 2016 | [MetaMind Neural Machine Translation System for WMT 2016](https://aclweb.org/anthology/W/W16/W16-2308.pdf) | | |
| ICML 2016 | [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576) | [GitHub](https://github.com/tscohen/GrouPy) | Chainer is not referred in the paper, but the authors kindly informed us.|
| CVPR 2017 Workshop | [Robocodes: Towards Generative Street Addresses from Satellite Imagery](https://research.fb.com/publications/robocodes-towards-generative-street-addresses-from-satellite-imagery/) |

<a name="blogs" />

## Official announcements

* [Chainer document](http://docs.chainer.org/en/latest/index.html) - An introduction to Chainer
* [Chainer blogs](http://chainer.org/blog/)

## Community

* [Introduction to Chainer: Neural Networks in Python](http://multithreaded.stitchfix.com/blog/2015/12/09/intro-to-chainer/)
* [The DIY Guide to Chainer](https://github.com/jxieeducation/DIY-Data-Science/blob/master/frameworks/chainer.md)
* [CHAINER CHARACTER EMBEDDINGS](http://dirko.github.io/Chainer-character-embeddings/)
* [A Fontastic Voyage: Generative Fonts with Adversarial Networks](http://multithreaded.stitchfix.com/blog/2016/02/02/a-fontastic-voyage/)


<a name="community" />

## Community
### Global

* [@ChainerOfficial on Twitter](https://twitter.com/ChainerOfficial)
* [Mailing List](https://groups.google.com/forum/#!forum/chainer)
* [Slack](http://bit.ly/chainer-slack)

### Japan

* [@ChainerJP on Twitter](https://twitter.com/ChainerJP)
* [Mailing List](https://groups.google.com/forum/#!forum/chainer-jp)
* [Slack](http://bit.ly/chainer-jp-slack)
* [connpass](https://chainer.connpass.com/)

<a name="books" />

## Books

* [Chainerによる実践深層学習](https://www.amazon.co.jp/dp/B01NBMKH21/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1) by 新納浩幸
