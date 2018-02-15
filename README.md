# Awesome Chainer [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

## What is Chainer?

Chainer is a flexible framework for neural networks. One of the major goals is flexibility, so it must enable us to write complex architectures simply and intuitively.

More info  [here](http://chainer.org/)

ref [Chainer.wiki](https://github.com/chainer/chainer/wiki) - Chainer Wiki

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

* [Chainer Tutorial Offical document](http://docs.chainer.org/en/latest/tutorial/index.html) - Chainer Tutorial Offical document

## Hands-on

*	[hido/chainer-handson](https://github.com/hido/chainer-handson/blob/master/chainer.ipynb) - Chainer Hands-on
*	[iwiwi/chainer-handson](https://github.com/iwiwi/chainer-handson/blob/master/chainer-ja.ipynb) - Chainer Hands-on(JP CPU only)
* [mitmul/chainer-handson](https://github.com/mitmul/chainer-handson) - Chainer Hands-on
* [mitmul/chainer-notebooks](https://github.com/mitmul/chainer-notebooks) - Chainer Jupyter Notebooks

<a name="github-projects" />

## Models/Projects

### Official Add-on Packages

* [ChainerRL](https://github.com/chainer/chainerrl) - ChainerRL is a deep reinforcement learning library built on top of Chainer.
* [ChainerCV](https://github.com/chainer/chainercv) - Versatile set of tools for Deep Learning based Computer Vision
* [ChainerMN](https://github.com/chainer/chainermn) - Scalable distributed deep learning with Chainer
* [ChainerUI](https://github.com/chainer/chainerui) - ChainerUI is a visualization and management tool for Chainer.


### Services using Chainer

* [PaintsChainer](https://github.com/pfnet/PaintsChainer) - Paints Chainer is line drawing colorizer using chainer.

<a name="#github-examples" />

# Chainer External Examples

## Preferred NetWorks Research

* [chainer-LSGAN](https://github.com/pfnet-research/chainer-LSGAN) - Least Squares Generative Adversarial Network implemented in Chainer
* [chainer-gogh](https://github.com/pfnet-research/chainer-gogh) - Implementation of "A neural algorithm of Artistic style"
* [chainer-graph-cnn](https://github.com/pfnet-research/chainer-graph-cnn) - Chainer implementation of 'Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering' (https://arxiv.org/abs/1606.09375)
* [chainer-segnet](https://github.com/pfnet-research/chainer-segnet) - SegNet implementation & experiments in Chainer
* [chainer-pix2pix](https://github.com/pfnet-research/chainer-pix2pix) - chainer implementation of pix2pix
* [chainer-ADDA](https://github.com/pfnet-research/chainer-ADDA) - Adversarial Discriminative Domain Adaptation
* [chainer-gan-lib](https://github.com/pfnet-research/chainer-gan-lib) - Various GANs implementation
* [tgan](https://github.com/pfnet-research/tgan) - Temporal Generative Adversarial Nets

### NLP

 * [odashi/chainer_examples](https://github.com/odashi/chainer_examples) - Machine Translation, word segmentation, and language model
 * [odashi/chainer_rnnlm.py](https://gist.github.com/odashi/0d6e259abcc14f2d2d28) - RNN Language Model
 * [odashi/chainer_encoder_decoder.py](https://gist.github.com/odashi/8d21f8fc23c075cd3042) - Neural Encoder-Decoder Machine Translation
 * [prajdabre/chainer_examples](https://github.com/prajdabre/chainer_examples/blob/master/chainer-1.5/LSTMVariants.py) - LSTM variants
 * [yusuketomoto/chainer-char-rnn](https://github.com/yusuketomoto/chainer-char-rnn) - Recurrent neural network (RNN)
 * [mlpnlp/mlpnlp-nmt](https://github.com/mlpnlp/mlpnlp-nmt) - LSTM encoder-decoder with attention mechanism
 * [unnonouno/chainer-memnn](https://github.com/unnonouno/chainer-memnn) - End-to-end memory networks
 * [jekbradbury/qrnn.py](http://metamind.io/research/new-neural-network-building-block-allows-faster-and-more-accurate-text-understanding/) - QRNN

### Computer Vision

 * [tscohen/GrouPy](https://github.com/tscohen/GrouPy) - Group Equivariant Convolutional Neural Networks
 * [mitmul/chainer-fast-rcnn](https://github.com/mitmul/chainer-fast-rcnn) - Fast R-CNN
 * [mitmul/chainer-faster-rcnn](https://github.com/mitmul/chainer-faster-rcnn) - Faster R-CNN
 * [mitmul/chainer-cifar10](https://github.com/mitmul/chainer-cifar10) - Cifar10
 * [mitmul/DeepPose](https://github.com/mitmul/deeppose) - Deep pose
 * [mitmul/chainer-conv-vis](https://github.com/mitmul/chainer-conv-vis) - Convolution Filter Visualization
 * [mitmul/chainer-imagenet-vgg](https://github.com/mitmul/chainer-imagenet-vgg) - VGG
 * [mitmul/chainer-segnet](https://github.com/mitmul/chainer-segnet) - SegNet
 * [mitmul/PSPNet](https://github.com/mitmul/chainer-pspnet) - Pyramid Scene Parsing Network
 * [apple2373/chainer-simple-fast-rnn](https://github.com/apple2373/chainer-simple-fast-rnn) - Fast R-CNN
 * [apple2373/chainer_stylenet](https://github.com/apple2373/chainer_stylenet) - StyleNet (A Neural Algorithm of Artistic Style)
 * [apple2373/chainer_caption_generation](https://github.com/apple2373/chainer_caption_generation) - Show and Tell
 * [mrkn/chainer-srcnn](https://github.com/mrkn/chainer-srcnn) - Image super-resolution
 * [Hi-king/chainer_superresolution](https://github.com/Hi-king/chainer_superresolution) - Image super-resolution
 * [darashi/chainer-example-overfeat-classify](https://github.com/darashi/chainer-example-overfeat-classify) - Overfeat
 * [RyotaKatoh/chainer-Variational-AutoEncoder](https://github.com/RyotaKatoh/chainer-Variational-AutoEncoder) - Variational autoencoder (VAE)
 * [yasunorikudo/chainer-ResNet](https://github.com/yasunorikudo/chainer-ResNet) - ResNet
 * [yasunorikudo/chainer-DenseNet](https://github.com/yasunorikudo/chainer-DenseNet) - DenseNet
 * [yasunorikudo/chainer-ResDrop](https://github.com/yasunorikudo/chainer-ResDrop) - ResDrop
 * [yusuketomoto/chainer-fast-neuralstyle](https://github.com/yusuketomoto/chainer-fast-neuralstyle) - Perceptual Losses for Real-Time Style Transfer and Super-Resolution
 * [rezoo/illustration2vec](https://github.com/rezoo/illustration2vec) - illustration2vec
 * [chainer-prednet](https://github.com/kunimasa-kawasaki/chainer-prednet) - Deep Predictive Coding Networks
 * [hillbig/binary_net](https://github.com/hillbig/binary_net) - BinaryNet
 * [stitchfix/fauxtograph](https://github.com/stitchfix/fauxtograph) - Variational Auto-Encoder (VAE), Generative Adversarial Nets (GAN), VAE-GAN
 * [rezoo/data.py](https://gist.github.com/rezoo/4e005611aaa4dad26697) - Generative Adversarial Nets (GAN)
 * [mattya/chainer-gogh](https://github.com/mattya/chainer-gogh) - StyleNet (A Neural Algorithm of Artistic Style)
 * [mattya/chainer-DCGAN](https://github.com/mattya/chainer-DCGAN) - Deep Convolutional Generative Adversarial Network (DCGAN)
 * [mattya/chainer-fluid](https://github.com/mattya/chainer-fluid) - Fluid simulation
 * [ktnyt/chainer_ca.py](https://gist.github.com/ktnyt/58e015dd9ff33049da5a) - Convolutional Autoencoder
 * [tochikuji/chainer-libDNN](https://github.com/tochikuji/chainer-libDNN/blob/master/examples/mnist/SdA.py) - Stacked Denoising Autoencoder
 * [masaki-y/ram](https://github.com/masaki-y/ram) - Recurrent Attention Model
 * [wkentaro/fcn](https://github.com/wkentaro/fcn) - Fully Convolutional Networks
 * [hvy/chainer-gan-denoising-feature-matching](https://github.com/hvy/chainer-gan-denoising-feature-matching) - Generative Adversarial Networks with Denoising Feature Matching
 * [hvy/chainer-visualization](https://github.com/hvy/chainer-visualization) - Visualizing and Understanding Convolutional Networks
 * [hvy/chainer-gan-trainer](https://github.com/hvy/chainer-gan-trainer) - Chainer GAN Trainer
 * [musyoku/wasserstein-gan](https://github.com/musyoku/wasserstein-gan) - WGAN
 * [weihua916/imsat](https://github.com/weihua916/imsat) - IMSAT
 * [Hakuyume/chainer-ssd](https://github.com/Hakuyume/chainer-ssd) - SSD
 * [leetenki/YOLOv2](https://github.com/leetenki/YOLOv2) - YOLOv2
 * [leetenki/YOLOtiny_v2](https://github.com/leetenki/YOLOtiny_v2_chainer) - YOLOtiny_v2
 * [yuyu2172/deformable-conv](https://github.com/yuyu2172/deformable-conv) - Deformable-conv
 * [dsanno/chainer-dfi](https://github.com/dsanno/chainer-dfi) - Deep Feature Interpolation for Image Content Changes
 * [dsanno/chainer-dfi](https://github.com/dsanno/chainer-dfi) - Deep Feature Interpolation for Image Content Changes

### Reinforcement Learning

 * [ugo-nama-kun/DQN-chainer](https://github.com/ugo-nama-kun/DQN-chainer) - Deep Q-Network (DQN)

### Generative models

* [crcrpar/chainer-VAE](https://github.com/crcrpar/chainer-VAE) - Variational AutoEncoder
* [musyoku/LSGAN](https://github.com/musyoku/LSGAN) - LSGAN
* [musyoku/began](https://github.com/musyoku/began) - BEGAN
* [musyoku/adversarial-autoencoder](https://github.com/musyoku/adversarial-autoencoder) - adversarial-autoencoder
* [musyoku/unrolled-gan](https://github.com/musyoku/unrolled-gan) - unrolled-ga
* [musyoku/improved-gan](https://github.com/musyoku/improved-gan)- improved-gan
* [musyoku/variational-autoencoder](https://github.com/musyoku/variational-autoencoder) - Semi-Supervised Learning with Deep Generative Models
* [musyoku/adgm](https://github.com/musyoku/adgm) - Auxiliary Deep Generative Models
* [musyoku/ddgm](https://github.com/musyoku/ddgm) - Deep Directed Generative Models with Energy-Based Probability Estimation
* [musyoku/minibatch_discrimination](https://github.com/musyoku/minibatch_discrimination) - Minibatch discrimination
* [musyoku/wavenet](https://github.com/musyoku/wavenet) - wavenet

### Unsupervised/Semi-supervised learning

* [musyoku/IMSAT](https://github.com/musyoku/IMSAT) - IMSAT
* [musyoku/vat](https://github.com/musyoku/vat) - VAT
* [musyoku/mnist-oneshot](https://github.com/musyoku/mnist-oneshot) - mnist-oneshot
* [mitmul/chainer-siamese](https://github.com/mitmul/chainer-siamese) - Siamese Network

### Others

* [mitmul/chainer-svm](https://github.com/mitmul/chainer-svm) - Support Vector Machine (SVM)

## Blog posts

* [Introduction to Chainer: Neural Networks in Python](http://multithreaded.stitchfix.com/blog/2015/12/09/intro-to-chainer/)
* [The DIY Guide to Chainer](https://github.com/jxieeducation/DIY-Data-Science/blob/master/frameworks/chainer.md)
* [CHAINER CHARACTER EMBEDDINGS](http://dirko.github.io/Chainer-character-embeddings/)
* [A Fontastic Voyage: Generative Fonts with Adversarial Networks](http://multithreaded.stitchfix.com/blog/2016/02/02/a-fontastic-voyage/)


## Tools and extensions

* [uei/Deel; A High level deep neural network description language](https://github.com/uei/deel)
* [uei/DEEPstation](https://libraries.io/github/uei/deepstation)
* [shi3z/chainer_imagenet_tools](https://github.com/shi3z/chainer_imagenet_tools)
* [lucidfrontier45/scikit-chainer](https://github.com/lucidfrontier45/scikit-chainer)
* [tochikuji/chainer-libDNN](https://github.com/tochikuji/chainer-libDNN)
* [musyoku/weight-normalization](https://github.com/musyoku/weight-normalization) - Weight Normalization Layer for Chainer
* [musyoku/chainer-sequential](https://github.com/musyoku/chainer-sequential) - chainer-sequential
* [musyoku/recurrent-batch-normalization](https://github.com/musyoku/recurrent-batch-normalization) - Recurrent Batch Normalization

<a name="video" />

## Videos

(in Japanese)

* [Chainer の Trainer 解説とNStepLSTM について](https://www.youtube.com/watch?v=ok_bvPKAEaM) Published on Mar 15, 2017
* [Chainer Meetup #04](https://www.youtube.com/watch?v=Fq5ZQ1ccG38&t=6837s) Published on Feb 23, 2017
* [1014：深層学習フレームワークChainerの導入と化合物活性予測への応用](https://www.youtube.com/watch?v=lM76gLQag4I&t=1211s) Published on Dec 2, 2015

<a name="papers" />

## Papers

* [GP-GAN: Towards Realistic High-Resolution Image Blending](https://arxiv.org/abs/1703.07195) 
  * Conference: arXiv only 
  * Codes:  [wuhuikai/GP-GAN](https://github.com/wuhuikai/GP-GAN) 
* [Temporal Generative Adversarial Nets](https://arxiv.org/abs/1611.06624)
  * Conference: arXiv only 
* [Reasoning with Memory Augmented Neural Networks for Language Comprehension](https://arxiv.org/abs/1610.06454)
  * Conference: arXiv only 
* [PMI Matrix Approximations with Applications to Neural Language Modeling](https://arxiv.org/abs/1609.01235)
  * Conference: arXiv only 
* [Neural Tree Indexers for Text Understanding](https://arxiv.org/abs/1607.04492)
  * Conference: arXiv only 
  * Codes:  [NTI](https://bitbucket.org/tsendeemts/nti/src) 
* [Neural Semantic Encoders](https://arxiv.org/abs/1607.04315)
  * Conference: arXiv only 
* [Networked Intelligence: Towards Autonomous Cyber Physical Systems](https://arxiv.org/abs/1606.04087)
  * Conference: arXiv only 
* [Modeling the dynamics of human brain activity with recurrent neural networks](https://arxiv.org/abs/1606.03071)
  * Conference: arXiv only 
* [A Deep-Learning Approach for Operation of an Automated Realtime Flare Forecast](https://arxiv.org/abs/1606.01587)
  * Conference: arXiv only 
* [Convolutional Neural Networks using Logarithmic Data Representation](https://arxiv.org/abs/1603.01025)
  * Conference: arXiv only 
* [context2vec: Learning Generic Context Embedding with Bidirectional LSTM](http://u.cs.biu.ac.il/%7Emelamuo/publications/context2vec_conll16.pdf)
  * Conference: CoNLL 2016 
* [Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec](https://arxiv.org/abs/1605.02019)
  * Conference:  CoNLL 2016  
* [Deep Impression: Audiovisual Deep Residual Networks for Multimodal Apparent Personality Trait Recognition](https://arxiv.org/abs/1609.05119) 
  * Conference:  ECCV 2016 Workshop  
  * comments: "3rd place in Looking at People ECCV Challenge"
* [Learning Joint Representations of Videos and Sentences with Web Image Search](https://arxiv.org/abs/1608.02367)
  * Conference:  ECCV 2016 Workshop  
* [Incorporating Discrete Translation Lexicons into Neural Machine Translation](https://arxiv.org/abs/1606.02006)
  * Conference: EMNLP 2016 
* [Controlling Output Length in Neural Encoder-Decoders](https://arxiv.org/abs/1609.09552)
  * Conference: EMNLP 2016 
* [Insertion Position Selection Model for Flexible Non-Terminals in Dependency Tree-to-Tree Machine Translation](http://www.aclweb.org/anthology/D16-1247)
  * Conference: EMNLP 2016 
* [Learning Representations Using Complex-Valued Nets](https://arxiv.org/abs/1511.06351)
  * Conference:  ICLR 2016  
* [Dynamic Coattention Networks For Qustion Answering](https://arxiv.org/abs/1611.01604)
  * Conference:  ICLR 2017 under review  
* [SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and < 0.5MB Model Size](https://arxiv.org/abs/1602.07360)
  * Conference:  ICLR 2017 under review  
* [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576)
  * Conference:  ICLR 2017 under review  
* [Steerable CNNs](https://arxiv.org/abs/1612.08498)
  * Conference:  ICLR 2017  
  * comments: Chainer is not referred in the paper, but the authors kindly informed us.
* [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://arxiv.org/abs/1606.00709)
  * Conference:  NIPS 2016 Workshop  
* [QSGD: Randomized Quantization for Communication-Optimal Stochastic Gradient Descent](https://arxiv.org/abs/1610.02132)
  * Conference:  OPT 2016  
* [Evaluation of Deep Learning based Pose Estimation for Sign Language Recognition](https://arxiv.org/abs/1602.09065)
  * Conference:  PETRA 2016  
* [Machine-learning Selection of Optical Transients in Subaru/Hyper Suprime-Cam Survey](https://arxiv.org/abs/1609.03249)
  * Conference:  PASJ 2016  
  * comments: PASJ: Publications of the Astronomical Society of Japan 
* [A Deep-Learning Approach for Operation of an Automated Realtime Flare Forecast](https://arxiv.org/abs/1606.01587)
  * Conference:  Space Weather 2016  
* [Dynamic Entity Representation with Max-pooling Improves Machine Reading](http://aclweb.org/anthology/N/N16/N16-1099.pdf)
  * Conference: NAACL 2016 
* [Feature-based Model versus Convolutional Neural Network for Stance Detection](http://aclweb.org/anthology/S/S16/S16-1065.pdf)
  * Conference: SemEval 2016 
* [Cross-Lingual Image Caption Generation](https://www.aclweb.org/anthology/P/P16/P16-1168.pdf)
  * Conference: ACL 2016 
* [Composing Distributed Representations of Relational Patterns](http://www.aclweb.org/anthology/P16-1215)
  * Conference: ACL 2016 
* [Generating Natural Language Descriptions for Semantic Representations of Human Brain Activity](https://www.aclweb.org/anthology/P/P16/P16-3004.pdf)
  * Conference: ACL 2016 
* [MetaMind Neural Machine Translation System for WMT 2016](https://aclweb.org/anthology/W/W16/W16-2308.pdf)
  * Conference: WMT 2016 
* [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576)
  * Conference: ICML 2016 
  * Codes:  [GitHub](https://github.com/tscohen/GrouPy) 
  * comments: Chainer is not referred in the paper, but the authors kindly informed us.
* [Robocodes: Towards Generative Street Addresses from Satellite Imagery](https://research.fb.com/publications/robocodes-towards-generative-street-addresses-from-satellite-imagery/)
  * Conference: CVPR 2017 Workshop 

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
