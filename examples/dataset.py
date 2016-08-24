#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gzip

import numpy as np
from six.moves import urllib
from six.moves import cPickle as pickle


def one_hot(x, depth):
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


def load_mnist_realval(path):
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
    return x_train, one_hot(t_train, n_y), x_valid, one_hot(t_valid, n_y), \
        x_test, one_hot(t_test, n_y)


def load_binary_mnist_realval(path):
    """
    Loads the binary real valued MNIST dataset.

    :param path: path to dataset file.
    :return: The MNIST dataset.
    """
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        load_mnist_realval(path)

    t_train = np.argmax(t_train)
    t_valid = np.argmax(t_valid)
    t_test = np.argmax(t_test)

    t_train = (t_train == 1).astype(np.float32)
    t_valid = (t_valid == 1).astype(np.float32)
    t_test = (t_test == 1).astype(np.float32)

    return x_train, t_train, x_valid, t_valid, x_test, t_test


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
