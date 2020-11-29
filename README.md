### Paper - [**BoMuDA: Boundless Multi-Source Domain Adaptive Segmentation in Unstructured Environments**](https://arxiv.org/abs/2010.03523)

Project Page - https://gamma.umd.edu/bomuda

Please cite our paper if you find it useful.

```
@article{kothandaraman2020bomuda,
  title={BoMuDA: Boundless Multi-Source Domain Adaptive Segmentation in Unconstrained Environments},
  author={Kothandaraman, Divya and Chandra, Rohan and Manocha, Dinesh},
  journal={arXiv preprint arXiv:2010.03523},
  year={2020}
}
```

<p align="center">
<img src="figures/cover.png" width="360">
</p>

Table of Contents
=================

  * [Paper - <a href="https://arxiv.org/abs/2010.03523" rel="nofollow"><strong>BoMuDA: Boundless Multi-Source Domain Adaptive Segmentation in Unstructured Environments</strong></a>](#paper---BoMuDA-Boundless-Multi-Source-Domain-Adaptive-Segmentation-in-Unstructured-Environments)
  * [**Repo Details and Contents**](#repo-details-and-contents)
     * [Code structure](#code-structure)
     * [Datasets](#datasets)
     * [Dependencies](#dependencies)
  * [**Our network**](#our-network)
  * [**Acknowledgements**](#acknowledgements)

## Repo Details and Contents
Python version: 3.7

### Code structure
#### Dataloaders <br>
The 'dataset' folder contains dataloaders for GTA5, BDD, CityScapes, Synscapes, and India Driving Dataset, and the corresponding train-test image splits. The dataloaders can be replicated for other datasets that the user may want to train on.
#### Models
The 'model' folder contains network architectures for DeepLab, Dilated Residual Networks, network used in step 2 of Alt-Inc algorithm (source3_concat.py), and the network used for the boundless module (openset_model.py)
#### Utils
Contains the cross entropy loss function
#### Training
Single-source domain adaptation models need to trained for initialization. This is done by the script 'train_singlesourceDA.py'. The paths to the source, and target domain datasets, along with the other paths can be set in lines 29-51 (Command line argument parsing can also be done). <br>

Step 1 (Alt-Inc) training - train_bddbase_multi3source_furtheriterations.py. The paths to the single-source models can be set in lines 242-251. The paths to the best source dataset, and target dataset, along with the other hyperparameters can be set in lines 29-51.

Step 2 (Alt-Inc) training - train_multi3source_combinedbddbase.py. The paths to the single-source models can be set in lines 159-161. The paths to the target dataset, along with the other hyperparameters can be set in lines 27-49.

Boundless Domain Adaptation Module - train_openset.py. The paths to the single-source models, and the closed-set multi-source model can be set in lines 172-176. The paths to the target dataset, along with the other hyperparameters can be set in lines 27-57.

#### Evaluation
eval_idd_BoMuDA.py - Evaluation script for India Driving Dataset, for the overall algorithm (Multi-source + Boundless)<br>

### Datasets
* [**India Driving Dataset**](https://idd.insaan.iiit.ac.in/) 
* [**CityScapes**](https://www.cityscapes-dataset.com/) 
* [**Berkeley Deep Drive**](https://bdd-data.berkeley.edu/) 
* [**GTA5**](https://download.visinf.tu-darmstadt.de/data/from_games/) 
* [**SynScapes**](https://7dlabs.com/synscapes-overview) 

### Dependencies
pytorch <br>
numpy <br>
scipy <br>
matplotlib <br>

## Our network

<p align="center">
<img src="overview.png">
</p>

## Acknowledgements

This code is heavily borrowed from [**AdaptSegNet**](https://github.com/wasidennis/AdaptSegNet)
