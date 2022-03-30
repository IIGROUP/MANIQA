# Multi-Dimension Attention Network for Image Quality Assessment

This repo is for NTIRE2022 Perceptual Image Quality Assessment Challenge Track 2 No-Reference competition.

## Dataset
Before running the codes, you should download the PIPAL datasets for [Training](https://drive.google.com/drive/folders/1G4fLeDcq6uQQmYdkjYUHhzyel4Pz81p-?usp=sharing) and [Validing](https://drive.google.com/drive/folders/1w0wFYHj8iQ8FgA9-YaKZLq7HAtykckXn).
Note that although we load the reference images, but we only use the distorted images as input for training and testing.

## Training & Testing & Ensemble
Training the MANNA model, run:
```
python train.py
```
For generating the ouput file, run:
```
python inference.py
```
For ensembling the model and generating the output file, run:
```
python ensemble.py
```
