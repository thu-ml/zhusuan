#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gzip

from six.moves import urllib
from six.moves import cPickle as pickle


def load_mnist_realval(path):
    """
    Loads the real valued MNIST dataset.

    :param path: path to dataset file.
    :return: The MNIST dataset.
    """
    def _download_mnist_realval(path):
        """
        Download the MNIST dataset if it is not present.
        """
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % url)
        urllib.request.urlretrieve(url, path)

    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        _download_mnist_realval(path)

    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    return x_train, t_train, x_valid, t_valid, x_test, t_test
