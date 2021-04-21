## High-order sampling strategy based on GraphSAGE

This package contains a PyTorch implementation of high-order sampling method on GraphSAGE.

## Environment settings

- python==3.6.8
- pytorch==1.0.0

## Basic Usage

**Main Parameters:**

```
--dataSet     The input graph dataset. (default: cora)
--agg_func    The aggregate function. (default: Mean aggregater)
--epochs      Number of epochs. (default: 50)
--b_sz        Batch size. (default: 20)
--seed        Random seed. (default: 824)
--unsup_loss  The loss function for unsupervised learning. ('margin' or 'normal', default: normal)
--config      Config file. (default: ./src/experiments.conf)
--cuda        Use GPU if declared.
```

**Learning Method**

The user can specify a learning method by --learn_method, 'sup' is for supervised learning, 'unsup' is for unsupervised learning, and 'plus_unsup' is for jointly learning the loss of supervised and unsupervised method.

**Example Usage**

To run the unsupervised model on Cuda:
```
python main.py --epochs 50 --dataSet reddit --learn_method unsup --cuda
```

