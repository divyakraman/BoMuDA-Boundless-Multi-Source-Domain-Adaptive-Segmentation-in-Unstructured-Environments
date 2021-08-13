### Paper - [**BoMuDANet: Unsupervised Adaptation for Visual Scene Understanding in Unstructured Driving Environments**](https://arxiv.org/abs/2010.03523)

Project Page - https://gamma.umd.edu/researchdirections/autonomousdriving/bomuda/ 

Watch the video [*here*](https://youtu.be/2TQ9lwohGos)

Please cite our paper if you find it useful.

```
@article{kothandaraman2020bomuda,
  title={BoMuDA: Boundless Multi-Source Domain Adaptive Segmentation in Unconstrained Environments},
  author={Kothandaraman, Divya and Chandra, Rohan and Manocha, Dinesh},
  journal={arXiv preprint arXiv:2010.03523},
  year={2020}
}
```

Table of Contents
=================
 * [Paper - <a href="link to paper" rel="nofollow"><strong>BoMuDANet: Unsupervised Adaptation for Visual Scene Understanding
in Unstructured Driving Environments</strong></a>](#paper---BoMuDANet-Unsupervised-Adaptation-for-Visual-Scene-Understanding-in-Unstructured-Driving-Environments)
  * [**Repo Details and Contents**](#repo-details-and-contents)
     * [Code structure](#code-structure)
     * [Training your own model](#training-your-own-model)
     * [Datasets](#datasets)
     * [Dependencies](#dependencies)
  * [**Our network**](#our-network)
  * [**Acknowledgements**](#acknowledgements)

## Repo Details and Contents
Python version: 3.7

### Code structure
#### Dataloaders <br>
|Dataset|Dataloader|List of images|
|-----|-----|-----|
|CityScapes|dataset/cityscapes.py|dataset/cityscapes_list|
|India Driving Dataset| dataset/cityscapes_fog.py | dataset/cityscapes_list (train_rain_fog, val_rain_fog) |
|Synthetic Rain | dataset/cityscapes_rain.py | dataset/cityscapes_list (train_rain_fog, val_rain_fog) |
|Synthetic Rain | dataset/cityscapes_rain.py | dataset/cityscapes_list (train_rain_fog, val_rain_fog) |
|Real Fog - Foggy Zurich| dataset/foggy_zurich/train(test).py | dataset/foggy_zurich/lists_file_names  |
|Real, Night Driving - Dark Zurich | dataset/dark_zurich/train(test).py | dataset/dark_zurich/lists_file_names  |
|Heterogeneous Real, Rain + Night - Raincouver | dataset/raincouver/raincouver.py | dataset/raincouver (train_rain_fog, val_rain_fog) |
|Heterogeneous Real, Berkeley Deep Drive | dataset/bdd/bdd_{train,val}.py | dataset/bdd/bdd_list |

### Our network
<p align="center">
<img src="ICCVW_Overview.png">
</p>

### Training your own model

**Stage 1**: Train networks for single source domain adaptation on various source-target pairs. <br>
```
python train_singlesourceDA.py
```
**Stage 2**: Use the trained single-source networks, and the corresponding domain discriminators for multi-source domain adaptation.
```
python train_bddbase_multi3source_furtheriterations.py
```
**Evaluation (closed-set DA)**: 
```
python eval_idd_BoMuDA.py
```
**Evaluation (open-set DA)**: 
```
python eval_idd_openset.py
```

### Datasets
* [**Clear weather: CityScapes**](https://www.cityscapes-dataset.com/) 
* [**India Driving Dataset**](https://idd.insaan.iiit.ac.in/)  
* [**GTA5**](https://download.visinf.tu-darmstadt.de/data/from_games/)
* [**SynScapes**](https://7dlabs.com/synscapes-overview)
* [**Berkeley Deep Drive**](https://bdd-data.berkeley.edu/) 


### Dependencies
PyTorch <br>
NumPy <br>
SciPy <br>
Matplotlib <br>

## Acknowledgements

This code is heavily borrowed from [**AdaptSegNet**](https://github.com/wasidennis/AdaptSegNet).
