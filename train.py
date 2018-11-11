#!/usr/bin/env python
import os
import argparse
import chainer
import chainer.functions as F
from chainer import training
from chainer.dataset import dataset_mixin
from chainer.training import extensions
from deepcluster import DeepClustering
from chainer import iterators
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class MnistDataset(dataset_mixin.DatasetMixin):
    def __init__(self, img):
        self.img = img

    def __len__(self):
        return len(self.img)

    def get_example(self, i):
        return i


class CalculateNMI(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, test_data, y, output_dir):
        self.model = model
        self.test_data = chainer.cuda.to_gpu(test_data)
        self.y = y
        self.output_dir = output_dir

    def __call__(self, trainer):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            pred, features = self.model.predict(self.test_data)
            features = chainer.cuda.to_cpu(features.data)
            pred = F.softmax(pred)
            pred = [np.argmax(pred_label) for pred_label in chainer.cuda.to_cpu(pred.data)]
            if self.model.prev_pred is None:
                self.model.prev_pred = pred
                prev_nmi_score = 0.0
            else:
                prev_nmi_score = normalized_mutual_info_score(self.model.prev_pred, pred)
                self.model.prev_pred = pred
        nmi_score = normalized_mutual_info_score(self.y, pred)
        chainer.report({'validation/NMI': nmi_score,
                        'validation/prevNMI': prev_nmi_score})

        test_images = chainer.cuda.to_cpu(self.test_data)
        for class_index in range(self.model.ncentroids):
            sample_images = test_images[(class_index == np.array(pred))][:6]
            for i in range(sample_images.shape[0]):
                plt.subplot(2, 3, i+1)
                plt.imshow(sample_images[i][0])
            filename = './' + self.output_dir + '/epoch_{.updater.epoch}' + '_predicted_class_' + str(class_index) + '.png'
            file_path = os.path.dirname(filename)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            plt.savefig(filename.format(trainer))


def kmeans_train(all_img):
    @training.make_extension(trigger=(1, 'epoch'))
    def _kmeans_train(trainer):
        model = trainer.updater.get_optimizer('main').target
        model.to_cpu()
        features = model.feature_extraction(all_img)
        model.kmeans_for_all(features, model.ncentroids, d=model.d)
        model.to_gpu()
    return _kmeans_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', default='mnist',
                        help='select dataset')
    parser.add_argument('--output_dir', default='result',
                        help='output directory path')
    parser.add_argument('--batchsize', default=256, type=int, help='image batchsize')
    parser.add_argument('--epoch', default=300, type=int, help='epoch number')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    args = parser.parse_args()

    if args.dataset_name == 'mnist':
        train, test = chainer.datasets.get_mnist(ndim=3)
        output_size = 10
    elif args.dataset_name == 'fashion-mnist':
        train, test = chainer.datasets.get_fashion_mnist(ndim=3)
        output_size = 10
    elif args.dataset_name == 'cifar10':
        train, test = chainer.datasets.get_cifar10(ndim=3)
        output_size = 10
    elif args.dataset_name == 'cifar100':
        train, test = chainer.datasets.get_cifar100(ndim=3)
        output_size = 100
    else:
        raise('Not Found Dataset')

    dataset = [train[i][0] for i in range(len(train))]
    test_dataset = np.array([test[i][0] for i in range(len(test))])
    y = np.array([test[i][1] for i in range(len(test))])

    print(len(dataset))
    print(y.shape[0])

    del train
    del test
    train_dataset = MnistDataset(dataset)
    dataset = np.array(dataset)

    model = DeepClustering(dataset, output_size)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
   
    train_iter = iterators.MultiprocessIterator(train_dataset, args.batchsize,
                                                repeat=True, shuffle=True,
                                                n_processes=4)
    updater = training.updaters.StandardUpdater(train_iter, optimizer,
                                                device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), 
                               out=args.output_dir)

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    snapshot_name = args.dataset_name + '_model_iter_{.updater.epoch}'
    trainer.extend(
        extensions.snapshot_object(model, snapshot_name),
        trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(kmeans_train(dataset))
    trainer.extend(
        extensions.PrintReport(['epoch', 'iteration', 'main/loss', 
                                'main/accuracy', 'validation/NMI', 
                                'validation/prevNMI', 'elapsed_time']),
        trigger=(1, 'epoch'))
    trainer.extend(
        extensions.PlotReport(['validation/NMI'], 'epoch',
                              file_name='NMI_' + args.dataset_name + '.png'))
    trainer.extend(
        extensions.PlotReport(['validation/prevNMI'], 'epoch',
                              file_name='prevNMI_' + args.dataset_name + '.png'))
    trainer.extend(CalculateNMI(model, test_dataset, y, args.output_dir), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
