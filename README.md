# CoinNet: Deep Ancient Roman Republican Coin Classification via Feature Fusion and Attention

This repository is for CoinNet: Deep Ancient Roman Republican Coin Classification via Feature Fusion and Attention (CoinNet) introduced in the following paper

Hafeez Anwar, [Saeed Anwar](https://saeed-anwar.github.io/), Sebastian Zambanini, [Fatih Porikli](https://www.porikli.com), "CoinNet: Deep Ancient Roman Republican Coin Classification via Feature Fusion and Attention", [arxiv](https://arxiv.org/abs/1908.09428) 

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
To this end, we present a novel network model, CoinNet, that employs compact bilinear pooling, residual groups, and feature attention layers. Furthermore, we gathered the largest and most diverse image dataset of the Roman Republican coins that contains more than 18,000 images belonging to 228 different reverse motifs. On this dataset, our model achieves a classification accuracy of more than 98% and outperforms the conventional bag-of-visual-words based approaches and more recent state-of-the-art deep learning methods. We also provide a detailed ablation study of our network and its generalization capability.


<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/CoinNet/tree/master/Figs/DatasetChallenge.png">
</p>
Variations in the anatomy of the reverse motifs due to the positions of the symbol (Red-dotted line border), main object(Blue-Solid line border), and legend(Orangedashed line border).

## Network
The following figure shows the architecture of our network
![Network](/Figs/Network.PNG)
CoinNet: Our model highlighting the Compact Bilinear Pooling, residual blocks, skip connections, and feature attention. The green and yellow cubes indicate the embedded features via CNN networks.


## Train
**Will be added later**

## Test
### Quick start
1. Download the trained models for our paper and place them in '/TestCode/experiment'.

    The real denoising model can be downloaded from [Google Drive](https://drive.google.com/open?id=1QxO6KFOVxaYYiwxliwngxhw_xCtInSHd) or [here](https://icedrive.net/0/e3Cb4ifYSl). The total size for all models is 5MB.

2. Cd to '/TestCode/code', run the following scripts.

    **You can use the following script to test the algorithm**

    ```bash
    #RIDNET
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model RIDNET --n_feats 64 --pre_train ../experiment/ridnet.pt --test_only --save_results --save 'RIDNET_RNI15' --testpath ../LR/LRBI/ --testset RNI15
    ```


## Results
**All the results for RIDNET can be downloaded from GoogleDrive from [SSID](https://drive.google.com/open?id=15peD5EvQ5eQmd-YOtEZLd9_D4oQwWT9e), [RNI15](https://drive.google.com/open?id=1PqLHY6okpD8BRU5mig0wrg-Xhx3i-16C) and [DnD](https://noise.visinf.tu-darmstadt.de/submission-detail). The size of the results is 65MB** 

### Quantitative Results
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/DnDTable.PNG">
</p>
The performance of state-of-the-art algorithms on widely used publicly available DnD dataset in terms of PSNR (in dB) and SSIM. The best results are highlighted in bold.

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/SSIDTable.PNG">
</p>
The quantitative results (in PSNR (dB)) for the SSID and Nam datasets.. The best results are presented in bold.

For more information, please refer to our [papar](https://arxiv.org/abs/1904.07396)

### Visual Results
![Visual_PSNR_DnD1](/Figs/DnD.PNG)
A real noisy example from DND dataset for comparison of our method against the state-of-the-art algorithms.

![Visual_PSNR_DnD2](/Figs/DnD2.PNG)
![Visual_PSNR_Dnd3](/Figs/DnD3.PNG)
Comparison on more samples from DnD. The sharpness of the edges on the objects and textures restored by our method is the best.

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/RNI15.PNG">
</p>
A real high noise example from RNI15 dataset. Our method is able to remove the noise in textured and smooth areas without introducing artifacts

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/SSID.PNG">
</p>
A challenging example from SSID dataset. Our method can remove noise and restore true colors

![Visual_PSNR_SSIM_BI](/Figs/SSID3.PNG)
![Visual_PSNR_SSIM_BI](/Figs/SSID2.PNG)
Few more examples from SSID dataset.

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
