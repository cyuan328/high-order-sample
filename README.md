## High-order sampling strategy based on GraphSAGE

This package contains a PyTorch implementation of high-order sampling method on GNNs.

## Environment settings

- python==3.6.2
- For the other packages, please refer to the requirements.txt.

## Basic Usage

To run the demo:
```sh run.sh```

All scripts of different models with parameters for Reddit, Cora, Citeseer and Pubmed are in `scripts` folder. You can reproduce the results by:
```
pip install -r requirements.txt
sh script/reddit_IncepGCN.sh
```

## Data
The data format is same as [GCN](https://github.com/tkipf/gcn). We provide three benchmark datasets as examples (see `data` folder). We use the public dataset splits provided by [Planetoid](https://github.com/kimiyoung/planetoid). The semi-supervised setting strictly follows [GCN](https://github.com/tkipf/gcn), while the full-supervised setting strictly follows [FastGCN](https://github.com/matenure/FastGCN) and [ASGCN](https://github.com/huangwb/AS-GCN). For the Reddit training, please first download the [Reddit](http://snap.stanford.edu/graphsage/reddit.zip).

