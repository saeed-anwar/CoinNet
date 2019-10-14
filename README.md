# CoinNet: Deep Ancient Roman Republican Coin Classification via Feature Fusion and Attention

This repository is for CoinNet: Deep Ancient Roman Republican Coin Classification via Feature Fusion and Attention (CoinNet) introduced in the following paper

Hafeez Anwar, [Saeed Anwar](https://saeed-anwar.github.io/), Sebastian Zambanini, [Fatih Porikli](https://porikli.com), "CoinNet: Deep Ancient Roman Republican Coin Classification via Feature Fusion and Attention", [arxiv](https://arxiv.org/abs/1908.09428) 

The model is built in PyTorch 0.4.0, PyTorch 0.4.1 and tested on Ubuntu 14.04/16.04 environment (Python3.6, CUDA9.0, cuDNN5.1). 


## Contents
1. [Introduction](#introduction)
2. [Network](#network)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
We perform classification of ancient Roman Republican coins via recognizing their reverse motifs where various objects, faces, scenes, animals, and buildings are minted along with legends. Most of these coins are eroded due to their age and varying degrees of preservation, thereby affecting their informative attributes for visual recognition. Changes in the positions of principal symbols on the reverse motifs also cause huge variations among the coin types. Lastly, in-plane orientations, uneven illumination, and a moderate background clutter further make the task of classification non-trivial and challenging.
To this end, we present a novel network model, CoinNet, that employs compact bilinear pooling, residual groups, and feature attention layers. Furthermore, we gathered the largest and most diverse image dataset of the Roman Republican coins that contains more than 18,000 images belonging to 228 different reverse motifs. On this dataset, our model achieves a classification accuracy of more than \textbf{98\%} and outperforms the conventional bag-of-visual-words based approaches and more recent state-of-the-art deep learning methods. We also provide a detailed ablation study of our network and its generalization capability.
