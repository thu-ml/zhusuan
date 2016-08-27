#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gzip

import numpy as np
from six.moves import urllib, range
from six.moves import cPickle as pickle


def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.

    :param x: 1-D Numpy array of type int.
    :return: 2-D Numpy array of type int.
    """
    ret = np.zeros((x.shape[0], depth))
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path)


def load_mnist_realval(path, one_hot=True):
    """
    Loads the real valued MNIST dataset.

    :param path: path to dataset file.
    :return: The MNIST dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://www.iro.umontreal.ca/~lisa/deep/data/mnist'
                         '/mnist.pkl.gz', path)

    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return x_train, t_transform(t_train), x_valid, t_transform(t_valid), \
        x_test, t_transform(t_test)


def load_binary_mnist_realval(path):
    """
    Loads real valued MNIST dataset for binary classification (Treat 0 & 2-9
    as 0).

    :param path: path to dataset file.
    :return: The binary labeled MNIST dataset.
    """
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        load_mnist_realval(path, one_hot=False)

    t_train = (t_train == 1).astype(np.float32)
    t_valid = (t_valid == 1).astype(np.float32)
    t_test = (t_test == 1).astype(np.float32)

    return x_train, t_train, x_valid, t_valid, x_test, t_test


def load_mnist_semi_supervised(path, one_hot=True):
    """
    Select 10 labeled data for each class and use all the other training data
    as unlabeled.

    :param path: path to dataset file.
    :return: The MNIST dataset for semi-supervised learning.
    """
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        load_mnist_realval(path, one_hot=False)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    t_train = np.hstack([t_train, t_valid])
    x_train_by_class = []
    t_train_by_class = []
    for i in range(10):
        indices = np.nonzero(t_train == i)[0]
        x_train_by_class.append(x_train[indices])
        t_train_by_class.append(t_train[indices])
    x_labeled = np.vstack([x[:10] for x in x_train_by_class])
    t_labeled = np.hstack([t[:10] for t in t_train_by_class])
    labeled_indices = np.arange(t_labeled.shape[0])
    np.random.shuffle(labeled_indices)
    x_labeled = x_labeled[labeled_indices]
    t_labeled = t_labeled[labeled_indices]
    x_unlabeled = np.vstack([x[10:] for x in x_train_by_class])
    np.random.shuffle(x_unlabeled)
    print(x_labeled.shape, t_labeled.shape, x_unlabeled.shape)
    return x_labeled, to_one_hot(t_labeled, 10), x_unlabeled, x_test, \
        to_one_hot(t_test, 10)


def load_german_credits(path, n_train):
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('https://archive.ics.uci.edu/ml/'
                         'machine-learning-databases/statlog/'
                         'german/german.data-numeric', path)

    n_dims = 24
    data = np.loadtxt(path)

    X_train = data[:n_train, :n_dims]
    y_train = data[:n_train, n_dims] - 1
    X_test = data[n_train:, :n_dims]
    y_test = data[n_train:, n_dims] - 1
    print('Finished reading data')

    return X_train, y_train, X_test, y_test
