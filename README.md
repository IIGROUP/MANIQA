# MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment

This repo is for NTIRE2022 Perceptual Image Quality Assessment Challenge Track 2 No-Reference competition.
We won first place in the competition and the codes have been released now.

![image.png](image/pipeline.png)

## Dataset
The training dataset is [PIPAL22](https://codalab.lisn.upsaclay.fr/competitions/1568#participate-get_data) and the validation dataset is [PIPAL21](https://competitions.codalab.org/competitions/28050#participate). We have conducted experiments on [LIVE](https://live.ece.utexas.edu/research/Quality/subjective.htm), [CSIQ](https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/), [TID2013](https://qualinet.github.io/databases/image/tampere_image_database_tid2013/) and [KADID-10K](http://database.mmsp-kn.de/kadid-10k-database.html) datasets. 

**NOTE:** Put the MOS label and the data python files into **data** folder. 
## Training MANIQA Model
```
# Training MANIQA model, run:
python train_maniqa.py
```
## Inference for [PIPAL22](https://codalab.lisn.upsaclay.fr/competitions/1568#participate-get_data) Validing and Testing
```
# Generating the ouput file, run:
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
## Acknowledgment
Our codes are revised from [anse3832](https://github.com/anse3832/MUSIQ).

