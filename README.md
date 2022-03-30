# Multi-Dimension Attention Network for Image Quality Assessment

This repo is for NTIRE2022 Perceptual Image Quality Assessment Challenge Track 2 No-Reference competition.

![image.png](image/pipeline.png)

## Dataset
Before running the codes, you should download the PIPAL datasets for [Training](https://drive.google.com/drive/folders/1G4fLeDcq6uQQmYdkjYUHhzyel4Pz81p-?usp=sharing) and [Validing](https://drive.google.com/drive/folders/1w0wFYHj8iQ8FgA9-YaKZLq7HAtykckXn).
Note that although we load the reference images, we only use the distorted images as input for training and testing.

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

## Environments & Requirements
- Platform: PyTorch 1.8.0
- Language: Python 3.7.9
- Ubuntu 18.04.6 LTS (GNU/Linux 5.4.0-104-generic x86\_64)
- CUDA Version 11.2
- GPU: NVIDIA GeForce RTX 3090 with 24GB memory
- torch, numpy, logging, tqdm, json, cv2, einops
