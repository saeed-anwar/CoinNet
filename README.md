# CoinNet: Deep Ancient Roman Republican Coin Classification via Feature Fusion and Attention

This repository is for CoinNet: Deep Ancient Roman Republican Coin Classification via Feature Fusion and Attention (CoinNet) introduced in the following paper

Hafeez Anwar, [Saeed Anwar](https://saeed-anwar.github.io/), Sebastian Zambanini, [Fatih Porikli](https://www.porikli.com), "CoinNet: Deep Ancient Roman Republican Coin Classification via Feature Fusion and Attention", [arxiv](https://arxiv.org/abs/1908.09428) 

The model is built in PyTorch 0.4.0, PyTorch 0.4.1 and tested on Ubuntu 14.04/16.04 environment (Python3.6, CUDA9.0, cuDNN5.1). 


## Contents
1. [Introduction](#introduction)
2. [Network](#network)
3. [Test](#test)
4. [Dataset](#dataset)
5. [Results](#results)
6. [Citation](#citation)

## Introduction
We perform classification of ancient Roman Republican coins via recognizing their reverse motifs where various objects, faces, scenes, animals, and buildings are minted along with legends. Most of these coins are eroded due to their age and varying degrees of preservation, thereby affecting their informative attributes for visual recognition. Changes in the positions of principal symbols on the reverse motifs also cause huge variations among the coin types. Lastly, in-plane orientations, uneven illumination, and a moderate background clutter further make the task of classification non-trivial and challenging.
To this end, we present a novel network model, CoinNet, that employs compact bilinear pooling, residual groups, and feature attention layers. Furthermore, we gathered the largest and most diverse image dataset of the Roman Republican coins that contains more than 18,000 images belonging to 228 different reverse motifs. On this dataset, our model achieves a classification accuracy of more than 98% and outperforms the conventional bag-of-visual-words based approaches and more recent state-of-the-art deep learning methods. We also provide a detailed ablation study of our network and its generalization capability.

<p align="center">
  <img width="450" src="https://github.com/saeed-anwar/CoinNet/blob/master/Figs/DatasetChallenge.png">
</p>
Variations in the anatomy of the reverse motifs due to the positions of the symbol (Red-dotted line border), main object(Blue-Solid line border), and legend(Orangedashed line border).

## Network
The following figure shows the architecture of our network

![Network](/Figs/Network.PNG)
Our model highlighting the Compact Bilinear Pooling, residual blocks, skip connections, and feature attention. The green and yellow cubes indicate the embedded features via CNN networks.

## Test
#### Quick start
1. Download the trained models for our paper and place them in '/TestCode/experiment'.

The real denoising model can be downloaded from [Google Drive](https://drive.google.com/open?id=1QxO6KFOVxaYYiwxliwngxhw_xCtInSHd) or [here](https://icedrive.net/0/e3Cb4ifYSl). The total size for all models is 5MB.

2. Cd to '/TestCode/code', run the following scripts.

    **You can use the following script to test the algorithm**

    ```bash
    #RIDNET
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model RIDNET --n_feats 64 --pre_train ../experiment/ridnet.pt --test_only --save_results --save 'RIDNET_RNI15' --testpath ../LR/LRBI/ --testset RNI15
    ```

## Dataset
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/CoinNet/blob/master/Figs/Dataset1.PNG">
</p>
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/CoinNet/blob/master/Figs/Dataset2.PNG">
</p>
Representative images: Samples images of the 100 classes that constitute the RRCD-Main.

## Results
**All the results for RIDNET can be downloaded from GoogleDrive from [SSID](https://drive.google.com/open?id=15peD5EvQ5eQmd-YOtEZLd9_D4oQwWT9e), [RNI15](https://drive.google.com/open?id=1PqLHY6okpD8BRU5mig0wrg-Xhx3i-16C) and [DnD](https://noise.visinf.tu-darmstadt.de/submission-detail). The size of the results is 65MB** 

#### Quantitative Results
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/CoinNet/blob/master/Figs/Table1.PNG">
</p>
Quantitative comparison: Comparison of our method with state-of-the-art methods on train-test split of 30%-70%. All results reported as top-1 mean accuracy on the test set

#### Ablation Study
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/CoinNet/blob/master/Figs/FeaturesEffect.PNG">
</p>
Input features effect: Comparison of different input features combinations to our CoinNet. Our network is robust to the change in the input features such as generated via ResNet50 (r50), DenseNet161 (d161) and Vgg19.

<p align="center">
  <img width="350" src="https://github.com/saeed-anwar/CoinNet/blob/master/Figs/VocabularyInfluence.PNG">
</p>
The effect of the vocabulary size on the classification performance for BoVWs and rectangular tiling.

<p align="center">
  <img width="350" src="https://github.com/saeed-anwar/CoinNet/blob/master/Figs/DisjointPerformance.PNG">
</p>
Accuracy on the unseen coin types for competing CNNs

#### Visual Results
<p align="center">
  <img width="450" src="https://github.com/saeed-anwar/CoinNet/blob/master/Figs/Attention.PNG">
</p>
Visualization results from Grad-CAM. The visualization is computed for the last convolutional outputs, and the ground-truth labels are shown on the left column the input images.

<p align="center">
  <img width="450" src="https://github.com/saeed-anwar/CoinNet/blob/master/Figs/Confidence.PNG">
</p>
The correctly classified images are represented with green circles while the wrongly classified ones are in red circles. In the first row, the confidence of the NasNet is always low although the model can classify correctly. The second shows that the confidence of the VGG, which is consistently high even for wrongly classified classes. The traditional classifiers as the CNN methods may be benefiting from the weights of ImageNet.

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{Anwar2019CoinNet,
  title={CoinNet: Deep Ancient Roman Republican Coin Classification via Feature Fusion and Attention},
  author={Anwar, Hafeez and Anwar, Saeed and Zambanini, Sebastian and Porikli, Fatih},
  journal={arXiv preprint arXiv:1908.09428},
  year={2019}
}
```
