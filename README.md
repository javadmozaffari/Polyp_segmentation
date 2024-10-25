# Polyp_segmentation
### Repository of the paper: ColonGen: An efficient polyp segmentation system for generalization improvement using a new comprehensive dataset.

#### This repository contains:
* Implementation of a novel deep learning model to segment the polyp images.
* The dataset files that were used in the study.
* Comprehensive dataset includes CVC-300, CVC-ColonDB, CVC-ClinicDB, Kvasir-seg, ETIS-Larib, PolypGen, Neopolyp-small, SUN-SEG, 

## Model implementation  
training:
```
python Train.py --train_path dataset/TrainDataset/ --test_path dataset/TestDataset/
```
    
    
testing:
```
python Test.py 
``` 
    
### Requirements  
* pytorch = 1.13.0  
* cuda = 11.6  
* mmcv= 1.5.0  
* timm   
* thop

## dataset
##### Download the training and testing dataset from this link: [Google Drive](https://drive.google.com/drive/folders/1UPLtqNFby6YEckr_GDPtK1VlCMbcSgUE?usp=share_link)

dataset folders:  
    
    |-- TrainDataset
    |   |-- images
    |   |-- masks
    |-- TestDataset
    |   |-- CVC-300
    |   |   |-- images
    |   |   |-- masks
    |   |-- CVC-ClinicDB
    |   |   |-- images
    |   |   |-- masks
    |   |-- CVC-ColonDB
    |   |   |-- images
    |   |   |-- masks
    |   |-- ETIS-LaribPolypDB
    |   |   |-- images
    |   |   |-- masks
    |   |-- Kvasir
    |       |-- images
    |       |-- masks



## Comprehensive dataset
##### Download the training and testing dataset from this link: [Google Drive]()
dataset folders:  
    
    |-- TrainDataset
    |   |-- images
    |   |-- masks
    |-- TestDataset
    |   |-- C5_polypgen
    |   |   |-- images
    |   |   |-- masks
    |   |-- C6_polypgen
    |   |   |-- images
    |   |   |-- masks
    |   |-- C10_ETIS
    |   |   |-- images
    |   |   |-- masks
    |   |-- C13_CVC-300
    |   |   |-- images
    |   |   |-- masks
    |   |-- C14_CVC_ColonDB
    |       |-- images
    |       |-- masks
    |   |-- C1_polypgen
    |       |-- images
    |       |-- masks
    |   |-- C2_polypgen
    |       |-- images
    |       |-- masks
    |   |-- C3_polypgen
    |       |-- images
    |       |-- masks
    |   |-- C4_polypgen
    |       |-- images
    |       |-- masks
    |   |-- C7_kvasir
    |       |-- images
    |       |-- masks
    |   |-- C8_CVC_Clinic
    |       |-- images
    |       |-- masks
    |   |-- C11_neopolyp_small
    |       |-- images
    |       |-- masks
    |   |-- C12_SUN-SEG
    |       |-- images
    |       |-- masks

## Acknowledgement
Part of our code is built based on [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT).

## Citation
Please cite our paper if you find the work useful:
