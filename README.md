# A Probabilistic Framework for Discovering New Intents

## Introduction
This repository provides the official PyTorch implementation of the research paper 'A Probabilistic Framework for Discovering New Intents'
### Dependencies 

We use anaconda to create python environment:
```
conda create --name python=3.9
```
Install all required libraries:
```
pip install -r requirements.txt
```

## Model Preparation
Get the pre-trained [BERT](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model and convert it into [Pytorch](https://huggingface.co/transformers/converting_tensorflow_models.html). 

Set the path of the uncased-bert model (parameter "bert_model" in init_parameter.py).

## Usage

Run the experiments by: 
```
sh scripts/run.sh
```





## Thanks
Some of the code was adapted from:
* https://github.com/thuiar/DeepAligned-Clustering
* https://github.com/fanolabs/NID_ACLARR2022


