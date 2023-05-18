# A Probabilistic Framework for Discovering New Intents

## Introduction
This repository provides the official PyTorch implementation of the research paper 'A Probabilistic Framework for Discovering New Intents'
### Dependencies 

We use anaconda to create python environment:
```
conda create --name python=3.6
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
You can change the parameters in the script. The selected parameters are as follows:
```
dataset: clinc | banking | stackoverflow
known_class_ratio: 0.25 | 0.5 | 0.75 (default)
cluster_num_factor: 1 (default) | 2 | 3 | 4 
```




## Thanks
Some of the code was adapted from:
* https://github.com/thuiar/DeepAligned-Clustering
* https://github.com/fanolabs/NID_ACLARR2022

If you are insterested in this work, and want to use the codes or results in this repository, please **star** this repository and **cite** by:
```
@article{Zhang_Xu_Lin_Lyu_2021, 
    title={Discovering New Intents with Deep Aligned Clustering}, 
    volume={35}, 
    number={16}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Zhang, Hanlei and Xu, Hua and Lin, Ting-En and Lyu, Rui}, 
    year={2021}, 
    month={May}, 
    pages={14365-14373}
}
```
### Acknowledgments
This paper is founded by seed fund of Tsinghua University (Department of Computer Science and Technology)- Siemens Ltd., China Joint Research Center for Industrial Intelligence and Internet of Things.
