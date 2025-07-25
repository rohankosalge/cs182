# CS W182/282A Self-Study
## May - July 2025

In this [course](https://cs182sp21.github.io/), I had the opportunity to:
- read several papers throughout the history of deep learning (from AlexNet to the Transformer, to the GAN, MAML, and more)
- critically understand the [lecture material](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVmKkQHucjPAoRtIJYt8a5A) and improve my mathematical maturity.
- apply my understanding through the homework assignments.<br>

Note: Homework 4 is not listed here. I was unable to install the necessary requirements for the RL-based assignment, as several modules/frameworks, including Python/Mac OS itself, have evolved in the last 4 years. I plan to resolve this issue in the near future.<br>

## Homework 1
**From scratch, I implemented the forward/backward passes for the layers of a FCNN and CNN. I also familiarized myself with the PyTorch API for the last part.**
*Lectures covered: 1-7*
*Time Spent: ~25 hours*<br>
There were 5 parts:
- FullyConnectedNets
	- affine layers
	- ReLU
	- loss layers (softmax, SVM)
	- constructing basic NN w/ above layers
	- [Very helpful source for understanding backprop](https://www.youtube.com/watch?v=VMj-3S1tku0&t=5673s) (Andrej Karpathy / YouTube)
- BatchNormalization
	- followed steps from the original [paper](https://proceedings.mlr.press/v37/ioffe15.pdf) (Sergey Ioffe and Christian Szegedy / ICML 2015)
	- [Very helpful source for understanding backprop for BN](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=5166s) (Andrej Karpathy / YouTube)
- Dropout
- ConvolutionalNetworks
	- convolutional layers
	- max-pooling layers
	- spatial batch normalization
- PyTorch
	- implemented a ConvNet structure using torch.nn.Module and torch.nn.Sequential
	- implemented hyperparameter tuning
	- on the open-ended challenge on CIFAR-10, I achieved a 77.11% accuracy against a benchmark accuracy of 70%.


## Homework 2
**In similar fashion, I implemented the layers of a vanilla RNN and LSTM, testing my models on the [Microsoft COCO Dataset](https://www.mscoco.org).**
*Lectures covered: 8-10*
*Time Spent: ~20 hours*
There were 4 parts:
- RNN_Captioning
	- RNN step layers
	- embedding layers
	- constructing basic RNN w/ these layers to perform image captioning
- LSTM_Captioning
	- LSTM step layers + embedding layers, same as RNN
	- the backprop method was trickier to implement, referred to this [source](https://kartik2112.medium.com/lstm-back-propagation-behind-the-scenes-andrew-ng-style-notations-7207b8606cb2) (Kartik Shenoy / Medium)
	- as part of the extra credit portion, I implemented an LSTM model to perform image captioning on COCO, achieving a BLEU score of ____ against a benchmark BLEU score of 0.3.
- NetworkVisualization
	- visualized saliency maps
	- generated fooling images, following this [paper](https://arxiv.org/pdf/1312.6199) (Szegedy et al. / ICLR 2014)
	- class visualization from noise, following this [paper](https://arxiv.org/pdf/1312.6034) (Simonyan et al. / ICLR Workshop 2014)
- StyleTransfer
	- calculated content loss + style loss, learned in lecture
	- reconstructed image from learned feature representation (feature inversion)


## Homework 3
*Lectures covered: 11-13*
*Time Spent: ~8 hours*
There were 2 parts + 1 experimental, extra credit part:
- Language Modeling
- Summarization
- Knowledge Distillation [EC]
