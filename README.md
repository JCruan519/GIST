# GIST: Improving Parameter Efficient Fine Tuning via Knowledge Interaction

This repo is the official implementation of the Gist framework. 




## Usage

### Install

- Create a conda virtual environment and activate it:

```bash
conda create -n gist python=3.8 -y
conda activate gist
```

- Install requirements:

```bash
pip install -r requirements.txt
```


### Data preparation

- VTAB-1K

You can follow ssf ("Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning") to download them. 

### Pre-trained model preparation

- For pre-trained ViT-B/16 on ImageNet-21K, the model weights will be automatically downloaded. You can also manually download them from [ViT](https://github.com/google-research/vision_transformer).



### Fine-tuning a pre-trained model via SSF

To fine-tune a pre-trained ViT model via `Adapter` within our GIST framework on VTAB-1K, run:

```bash
bash train_scripts/vit/train_vtab.sh
```



### Acknowledgement
The code is built upon [ssf](https://github.com/dongzelian/SSF).
