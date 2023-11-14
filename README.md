# Distinguishing Latent Interaction Types from Implicit Feedbacks for Recommendation

This is the PyTorch Implementation for the paper [RAGM](https://www.sciencedirect.com/science/article/pii/S0020025523014196):

> Lingyun Lu, Bang Wang, Zizhuo Zhang and Shenghao Liu. Distinguishing Latent Interaction Types from Implicit Feedbacks for Recommendation.

## Introduction

Relation-Aware Neural Model (RAGM) is a recommendation framework based on implicit feedbacks, which explicitly distinguishes the information of different interaction relations for user preference modeling and a
Relation-Aware Graph Nerual Network (RAGNN) is designed for user and item encoding.

## Citation 

If you want to use our codes and datasets in your research, please cite:

```
xxx
```

## Environment Requirement

The code has been tested running under Python 3.8.0. The required packages are as follows:

- pytorch==1.10.1
- networkx==2.5.1
- numpy==1.22.4
- pandas==1.4.3
- scikit-learn==1.1.1
- scipy==1.7.0
- torch==1.9.0
- torch-cluster==1.5.9
- torch-scatter==2.0.9
- torch-sparse==0.6.12

## Usage

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). 
We take MovieLens dataset as example:

- MovieLens dataset

```
python main.py --dataset MovieLens --lr 0.0001 --n_factors 3 --context_hops 2
```


## Dataset

We provide three processed datasets: Last-FM and MovieLens.

- You can find the full version of recommendation datasets via [Yelp](https://www.yelp.com/dataset/challenge), [Douban Movie](http://www.shichuan.org/HIN_dataset.html) and [MovieLens](https://grouplens.org/datasets/movielens/).
- We follow the previous study to preprocess the datasets.

|                       |               |    Yelp | Douban Movie | MovieLens 1M |
| :-------------------: | :------------ |--------:|-------------:|--------------|
| User-Item Interaction | #Users        |  14,085 |        3,022 | 6,040        |
|                       | #Items        |  14,037 |        6,971 | 3,706        |
|                       | #Interactions | 194,255 |      195,493 | 1,000209     |



