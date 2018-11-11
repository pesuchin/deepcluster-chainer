import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import cuda, optimizers, serializers, Variable
import math
import six
from chainer.links.caffe.protobuf3 import caffe_pb2 as caffe_pb
import faiss
from scipy.ndimage.interpolation import rotate


# 参考: https://github.com/chainer/chainer/blob/master/examples/imagenet/alex.py
class DeepClustering(chainer.Chain):
    def __init__(self, all_img, output_size, train=True):
        super(DeepClustering, self).__init__()
        with self.init_scope():
            self.batchnorm = L.BatchNormalization(96)
            self.batchnorm2 = L.BatchNormalization(256)
            self.conv1 = L.Convolution2D(None,  96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256,  5, pad=2)
            self.conv3 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv4 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv5 = L.Convolution2D(None, 256,  3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, output_size)
        if train:
            x = chainer.cuda.to_cpu(all_img)
            features = self.feature_extraction(x)
            self.ncentroids = output_size
            self.d = 256
            self.kmeans_for_all(features, self.ncentroids, d=self.d)
            self.all_img = chainer.cuda.to_gpu(all_img)
            self.prev_pred = None

    def __call__(self, i):
        x = self.all_img[i]
        # data augumentation
        if np.random.random() > 0.5:
            x = F.flip(x, 2)

        pred, _ = self.predict(x)

        # P.6 trival parametrization solution
        pseudo_labels = chainer.cuda.to_cpu(self.pseudo_labels[i])
        pseudo_labels = np.append(pseudo_labels, [self.ncentroids])
        class_weight = np.bincount(pseudo_labels)[:self.ncentroids]
        class_weight[class_weight == 0] = 1.0
        class_weight = (1.0 / class_weight).astype('float32')
        class_weight = chainer.cuda.to_gpu(class_weight)

        self.loss = F.softmax_cross_entropy(pred, self.pseudo_labels[i], 
                                            class_weight=class_weight)

        chainer.report({'loss': self.loss,
                       'accuracy': F.accuracy(pred, self.pseudo_labels[i])},
                        self)

        return self.loss

    def predict(self, x):
        feature = F.max_pooling_2d(
            self.batchnorm(F.relu(self.conv1(x))), 3, stride=2)
        feature = F.max_pooling_2d(
            self.batchnorm2(F.relu(self.conv2(feature))), 3, stride=2)
        feature = F.relu(self.conv3(feature))
        h = F.relu(self.conv4(feature))
        h = F.relu(self.conv5(feature))
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = F.dropout(F.relu(self.fc6(h)), ratio=.5)
        h = F.dropout(F.relu(self.fc7(h)), ratio=.5)
        self.pred = self.fc8(h)
        return self.pred, feature

    def feature_extraction(self, x):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            feature = F.max_pooling_2d(F.local_response_normalization(
                                       F.relu(self.conv1(x))), 3, stride=2)
            feature = F.max_pooling_2d(F.local_response_normalization(
                F.relu(self.conv2(feature))), 3, stride=2)
            feature = F.relu(self.conv3(feature))
            feature = F.relu(self.conv4(feature))
        return feature

    def apply_pca(self, feature, out_dim):
        data_size, C, W, H = feature.data.shape
        feature_size = C * W * H
        mt = chainer.cuda.to_cpu(feature.data.reshape([data_size, feature_size]))
        self.mat = faiss.PCAMatrix(feature_size, out_dim)
        self.mat.train(mt)
        assert self.mat.is_trained
        x = self.mat.apply_py(mt)

        # L2 normalization
        row_sums = np.linalg.norm(x, axis=1)
        x = x / row_sums[:, np.newaxis]

        return x

    def run_kmeans(self, x, nmb_clusters):
        _, d = x.shape
        # faiss implementation of k-means
        clus = faiss.Clustering(d, nmb_clusters)
        clus.niter = 20
        # clus.max_points_per_centroid = 10000000
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        self.index = faiss.GpuIndexFlatL2(res, d, flat_config)

        # perform the training
        clus.train(x, self.index)
        _, I = self.index.search(x, 1)
        losses = faiss.vector_to_array(clus.obj)
        # print('k-means loss evolution: {0}'.format(losses))
        return I.ravel(), losses[-1]

    def kmeans_for_all(self, feature, k, d=256):
        # PCA size->256
        # also does a random rotation after the reduction (the 4th argument)
        x = self.apply_pca(feature, d)

        I, self.clustering_loss = self.run_kmeans(x, k)

        # assign pseudo-labels
        self.pseudo_labels = chainer.cuda.to_gpu(np.array(I))
