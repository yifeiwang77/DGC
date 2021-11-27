# Decoupled Graph Convolution (DGC)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Authors: [Yifei Wang](https://scholar.google.com.tw/citations?user=sNL8SSoAAAAJ&hl=en), [Yisen Wang](https://yisenwang.github.io/), [Jiansheng Yang](http://english.math.pku.edu.cn/peoplefaculty/64.html), [Zhouchen Lin](https://zhouchenlin.github.io/)

## Overview
This repo contains an example implementation of the **Decoupled Graph Convolution (DGC)** model, described in the NeurIPS 2021 paper [Dissecting the Diffusion Process in Linear Graph Convolutional Networks](https://arxiv.org/abs/2102.10739).

DGC, similar to [SGC](https://github.com/Tiiiger/SGC), removes the nonlinearities and collapes the weight matrices in Graph Convolutional Networks (GCNs) and is essentially a **linear GCN**. 

Motivated by the dissection of SGC's limitations from a continuous perspective, DGC further ***decouples*** the  terminal time $T\in\mathbb{R}$ and propagation steps $K\in\mathbb{N}$ as two free hyperparameters. 
In this way, DGC overcomes SGC's limitations and **improves SGC by a large margin**, making it even comparable to state-of-the-art nonlinear GCNs. Meanwhile, as a linear GCN, DGC is very memory-efficient and saves much training time. 

This repo contains the implementation of DGC for citation networks (Cora, Citeseer, and Pubmed) and the performance is shown below. All experiments are conducted with a single NVIDIA GTX 1080ti GPU.

Dataset | Acc (%) | $T$ (diffusion time) | $K$ (steps) | Training Time
:------:|:-----------:|:-------:|:-----------:|:-----------:|
Cora    |83.3 ± 0.0 | 5.27  | 250 | 0.37s 
Citeseer|73.3 ± 0.1| 3.78 | 300 | 0.86s 
Pubmed  |80.3 ± 0.1| 6.05 | 900 | 2.35s

In particular, DGC could complete Cora/Citeseer training with <1s, and training on a large graph (Pubmed) with 900 steps only takes 2.35s.

## Dependencies
Our implementation works with PyTorch>=1.0.0 and you can install other dependencies with

``$ pip install -r requirement.txt``

## Usage
We provide the citation network datasets under `data/`, which corresponds to [the public data splits](https://github.com/tkipf/gcn/tree/master/gcn/data).

To train DGC on the citation networks, simply run ``main.py`` with the following commands,
```
python main.py --dataset <cora, citeseer, or pubmed> --T <T value> --K <K value>
```
To reproduce the results above, run ```sh run_citation.sh```. It will automatically run 10 trials for each experiment and report the mean and standard deviation. 

---
If you find this repo useful, please cite: 
```
@InProceedings{wang2021dgc,
  title = 	 {Dissecting the Diffusion Process in Linear Graph Convolutional Networks},
  author = 	 {Wang, Yifei and Wang, Yisen and Yang, Jiansheng and Lin, Zhouchen},
  booktitle = {Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021)},
  year = 	 {2021}
}
```

## Acknowledgement
This repo borrows a lot from [SGC](https://github.com/Tiiiger/SGC), which is modified from [pygcn](https://github.com/tkipf/pygcn), and [FastGCN](https://github.com/matenure/FastGCN).
