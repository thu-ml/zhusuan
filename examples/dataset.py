#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gzip
import tarfile

import numpy as np
from six.moves import urllib, range
from six.moves import cPickle as pickle


def standardize(data_train, data_test):
    """
    Standardize a dataset to have zero mean and unit standard deviation.

    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.

    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0)
    std[std == 0] = 1
    mean = np.mean(data_train, 0)
    data_train_standardized \
        = (data_train - np.full(data_train.shape, mean, dtype='float32')) / \
        np.full(data_train.shape, std, dtype='float32')
    data_test_standardized \
        = (data_test - np.full(data_test.shape, mean, dtype='float32')) / \
        np.full(data_test.shape, std, dtype='float32')
    return data_train_standardized, data_test_standardized, mean, std


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

    :param path: Path to dataset file.
    :param one_hot: Use one-hot representation for the labels.
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


def load_mnist_semi_supervised(path, one_hot=True, seed=123456):
    """
    Select 10 labeled data for each class and use all the other training data
    as unlabeled.

    :param path: path to dataset file.
    :param one_hot: Use one-hot representation for the labels.
    :param seed: Random seed for selecting labeled data.

    :return: The MNIST dataset for semi-supervised learning.
    """
    rng = np.random.RandomState(seed=seed)
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
    x_labeled = []
    t_labeled = []
    for i in range(10):
        indices = np.arange(x_train_by_class[i].shape[0])
        rng.shuffle(indices)
        x_labeled.append(x_train_by_class[i][indices[:10]])
        t_labeled.append(t_train_by_class[i][indices[:10]])
    x_labeled = np.vstack(x_labeled)
    t_labeled = np.hstack(t_labeled)
    x_unlabeled = x_train
    np.random.shuffle(x_unlabeled)
    t_transform = (lambda x: to_one_hot(x, 10)) if one_hot else (lambda x: x)
    return x_labeled, t_transform(t_labeled), x_unlabeled, x_test, \
        t_transform(t_test)


def load_cifar10(path, normalize=True, dequantify=False, one_hot=True):
    """
    Loads the cifar10 dataset.

    :param path: path to dataset file.
    :param normalize: normalize the x data to the range [0, 1].
    :param dequantify: Add uniform noise to dequantify the data following (
        Uria, 2013).
    :param one_hot: Use one-hot representation for the labels.

    :return: The cifar10 dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset(
            'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', path)

    data_dir = os.path.dirname(path)
    batch_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.isfile(os.path.join(batch_dir, 'data_batch_5')):
        with tarfile.open(path) as tar:
            tar.extractall(data_dir)

    train_x, train_y = [], []
    for i in range(1, 6):
        batch_file = os.path.join(batch_dir, 'data_batch_' + str(i))
        with open(batch_file, 'r') as f:
            data = pickle.load(f)
            train_x.append(data['data'])
            train_y.append(data['labels'])
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)

    test_batch_file = os.path.join(batch_dir, 'test_batch')
    with open(test_batch_file, 'r') as f:
        data = pickle.load(f)
        test_x = data['data']
        test_y = np.asarray(data['labels'])

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    print(train_x.min(), train_x.max())
    print(test_x.min(), test_x.max())
    if dequantify:
        train_x += np.random.uniform(0, 1,
                                     size=train_x.shape).astype('float32')
        test_x += np.random.uniform(0, 1, size=test_x.shape).astype('float32')
    if normalize:
        train_x = train_x / 256
        test_x = test_x / 256

    train_x = train_x.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_x = test_x.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    t_transform = (lambda x: to_one_hot(x, 10)) if one_hot else (lambda x: x)
    return train_x, t_transform(train_y), test_x, t_transform(test_y)


def load_cifar10_semi_supervised(path, normalize=True, dequantify=False,
                                 one_hot=True, seed=123456):
    """
    Select 400 labeled data for each class and use all the other training data
    as unlabeled.

    :param path: path to dataset file.
    :param one_hot: Use one-hot representation for the labels.
    :param seed: Random seed for selecting labeled data.

    :return: The cifar10 dataset for semi-supervised learning.
    """
    rng = np.random.RandomState(seed=seed)
    x_train, t_train, x_test, t_test = load_cifar10(
        path, normalize=normalize, dequantify=dequantify, one_hot=False)
    x_train_by_class = []
    t_train_by_class = []
    for i in range(10):
        indices = np.nonzero(t_train == i)[0]
        x_train_by_class.append(x_train[indices])
        t_train_by_class.append(t_train[indices])
    x_labeled = []
    t_labeled = []
    for i in range(10):
        indices = np.arange(x_train_by_class[i].shape[0])
        rng.shuffle(indices)
        x_labeled.append(x_train_by_class[i][indices[:400]])
        t_labeled.append(t_train_by_class[i][indices[:400]])
    x_labeled = np.vstack(x_labeled)
    t_labeled = np.hstack(t_labeled)
    x_unlabeled = x_train
    np.random.shuffle(x_unlabeled)
    t_transform = (lambda x: to_one_hot(x, 10)) if one_hot else (lambda x: x)
    return x_labeled, t_transform(t_labeled), x_unlabeled, x_test, \
        t_transform(t_test)


def load_uci_german_credits(path, n_train):
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


def load_uci_boston_housing(path):
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://archive.ics.uci.edu/ml/'
                         'machine-learning-databases/housing/housing.data',
                         path)

    data = np.loadtxt(path)
    permutation = np.random.choice(np.arange(data.shape[0]),
                                   data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.8))
    size_test = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0: size_train]
    index_test = permutation[size_train:size_test]
    index_val = permutation[size_test:]

    X_train, y_train = data[index_train, :-1], data[index_train, -1]
    X_val, y_val = data[index_val, :-1], data[index_val, -1]
    X_test, y_test = data[index_test, :-1], data[index_test, -1]

    return X_train, y_train, X_val, y_val, X_test, y_test
