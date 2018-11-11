DeepCluster-chainer
====


## Description

This is DeepCluster for chainer. But this implementation is not fixed.

The paper of this wonderful model is below.

[Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520)

- [x] Evaluator
- [ ] Feature that select your data 
- [ ] Model implementation
- [ ] hyper-parameter tuning
- [ ] Prepare the Demo

## Demo

## Requirement

## Usage

```
$ python train.py [dataset_name] [--output_dir 'result'] [--batchsize 256] [--epoch 300] [--gpu 0]

for example:
$ python train.py mnist --output_dir='result/mnist/' --epoch 50
```

support dataset is bellow:

- mnist
- fashion-mnist
- CIFAR 10
- CIFAR 100

## Install

## Contribution

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[pesuchin](https://github.com/pesuchin)
