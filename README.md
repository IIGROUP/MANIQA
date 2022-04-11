# Multi-Dimension Attention Network for Image Quality Assessment

This repo is for NTIRE2022 Perceptual Image Quality Assessment Challenge Track 2 No-Reference competition.

![image.png](image/pipeline.png)

## Dataset
The dataset we use are [PIPAL22](https://codalab.lisn.upsaclay.fr/competitions/1568#participate-get_data) for training and validing. We also use [LIVE](https://live.ece.utexas.edu/research/Quality/subjective.htm), [CSIQ](https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/), [TID2013](https://qualinet.github.io/databases/image/tampere_image_database_tid2013/) and [KADID-10K](http://database.mmsp-kn.de/kadid-10k-database.html).
## Training
Training the MANIQA model, run:
```
python train.py
```
## Inference
For generating the ouput file, run:
```
python inference.py
```
## Environments & Requirements
- Platform: PyTorch 1.8.0
- Language: Python 3.7.9
- Ubuntu 18.04.6 LTS (GNU/Linux 5.4.0-104-generic x86\_64)
- CUDA Version 11.2
- GPU: NVIDIA GeForce RTX 3090 with 24GB memory

 Python requirements can installed by:
```
pip install -r requirements.txt
```
