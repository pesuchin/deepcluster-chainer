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
from chainer.datasets import TransformDataset
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class MnistDataset(dataset_mixin.DatasetMixin):
    def __init__(self, img):
        self.img = img

    def __len__(self):
        return len(self.img)

    def get_example(self, i):
        return self.img[i], i


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
            pred = self.model.predict(self.test_data)
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

        filename = './' + self.output_dir + '/epoch_{.updater.epoch}_conv.png'
        file_path = os.path.dirname(filename)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        test_images = chainer.cuda.to_cpu(self.test_data)
        # for class_index in range(self.model.ncentroids):
        for class_index in range(10):
            fig = plt.figure()
            sample_images = test_images[(class_index == np.array(pred))][:6]
            for i in range(sample_images.shape[0]):
                fig.add_subplot(2, 3, i+1)
                if sample_images[i].shape == (1, 28, 28):
                    plt.imshow(sample_images[i][0])
                else:
                    plt.imshow(sample_images[i].transpose(1, 2, 0))
            filename = './' + self.output_dir + '/epoch_{.updater.epoch}' + '_predicted_class_' + str(class_index) + '.png'
            plt.savefig(filename.format(trainer))


def kmeans_train(all_img):
    @training.make_extension(trigger=(1, 'epoch'))
    def _kmeans_train(trainer):
        model = trainer.updater.get_optimizer('main').target
        model.to_cpu()
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            features = model.feature_extraction(all_img)
        model.kmeans_for_all(features, model.ncentroids, d=model.d)
        model.to_gpu()
    return _kmeans_train


def dataset_preprocess(train, test):
    dataset = [train[i][0] for i in range(len(train))]
    y = np.array([train[i][1] for i in range(len(train))])
    test_dataset = np.array([test[i][0] for i in range(len(test))])
    test_y = np.array([test[i][1] for i in range(len(test))])

    print(len(dataset))
    print(y.shape[0])
    return dataset, y, test_dataset, test_y


def cutout(image_origin, mask_size):
    image = np.copy(image_origin)
    mask_value = image.mean()

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    image[top:bottom, left:right, :].fill(mask_value)
    return image


def resize_image(img, size=(224, 224)):
    w, h = img.size
    img = img.resize((int(w * (size[1] / h)), size[1]))
    img = np.array(img, dtype=np.float32)
    ch, h, w = img.shape
    offset_w = (w - size[0]) // 2
    img = img[:, offset_w:offset_w+size[0], :]
    return img


def center_crop(img, size, return_param=False, copy=False):
    _, H, W = img.shape
    oH, oW = size
    if oH > H or oW > W:
        raise ValueError('shape of image needs to be larger than size')

    y_offset = int(round((H - oH) / 2.))
    x_offset = int(round((W - oW) / 2.))

    y_slice = slice(y_offset, y_offset + oH)
    x_slice = slice(x_offset, x_offset + oW)

    img = img[:, y_slice, x_slice]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_slice': y_slice, 'x_slice': x_slice}
    else:
        return img


def transform(x, angle_range=(0, 30)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(x, mean, std):
        t = (t - m) / s
    # x = center_crop(x, (224, 224))
    x = x.transpose(1, 2, 0)
    # h, w, _ = x.shape

    # angle = np.random.randint(*angle_range)
    # if np.random.rand() > 0.5:
    #     x = rotate(x, angle)
    #     x = imresize(x, (h, w))

    # if np.random.rand() > 0.5:
    #     cutout(x, 5)
    
    # x_offset = np.random.randint(4)
    # y_offset = np.random.randint(4)
    # x = x[y_offset:y_offset + h - 4,
    #       x_offset:x_offset + w - 4]
    if np.random.rand() > 0.5:
        x = np.fliplr(x)

    x = x.transpose(2, 0, 1)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', default='mnist',
                        help='select dataset')
    parser.add_argument('--output_dir', default='result',
                        help='output directory path')
    parser.add_argument('--batchsize', default=256, type=int, help='image batchsize')
    parser.add_argument('--epoch', default=300, type=int, help='epoch number')
    parser.add_argument('--fully_output_size', default=4096, type=int, help='fully connected unit size')
    parser.add_argument('--pca_dim', default=128, type=int, help='pca output dims')
    parser.add_argument('--verbose', default=False, type=bool, help='print hidden size')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    args = parser.parse_args()

    if args.pca_dim < 0:
        use_pca = False
    else:
        use_pca = True

    if args.dataset_name == 'mnist':
        train, test = chainer.datasets.get_mnist(ndim=3)
        output_size = 10
        sobel = False
        dataset, y, test_dataset, test_y = dataset_preprocess(train, test)
    elif args.dataset_name == 'fashion-mnist':
        train, test = chainer.datasets.get_fashion_mnist(ndim=3)
        output_size = 10
        sobel = False
        dataset, y, test_dataset, test_y = dataset_preprocess(train, test)
    elif args.dataset_name == 'cifar10':
        train, test = chainer.datasets.get_cifar10(ndim=3)
        output_size = 10
        sobel = True
        dataset, y, test_dataset, test_y = dataset_preprocess(train, test)
    elif args.dataset_name == 'cifar100':
        train, test = chainer.datasets.get_cifar100(ndim=3)
        output_size = 100
        sobel = True
        dataset, y, test_dataset, test_y = dataset_preprocess(train, test)
    else:
        raise('Not Found Dataset')

    train_dataset = TransformDataset(dataset, transform)
    train_dataset = MnistDataset(train_dataset)
    dataset = np.array(dataset)

    print('fully_output_size:', args.fully_output_size)
    print('pca_dim:', args.pca_dim)
    print('use_pca:', use_pca)
    print('input shape:', train[0][0].shape)

    model = DeepClustering(dataset, args.pca_dim, args.fully_output_size, output_size, 
                           verbose=args.verbose, sobel=sobel, use_pca=use_pca)
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
    trainer.extend(kmeans_train(dataset))

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    snapshot_name = args.dataset_name + '_model_iter_{.updater.epoch}'
    trainer.extend(
        extensions.snapshot_object(model, snapshot_name),
        trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
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
    trainer.extend(CalculateNMI(model, dataset, y, args.output_dir), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
