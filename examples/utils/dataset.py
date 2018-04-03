#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gzip
import tarfile
import zipfile
import progressbar
import math

import numpy as np
import six
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
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    data_test_standardized = (data_test - mean) / std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return data_train_standardized, data_test_standardized, mean, std


def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.

    :param x: 1-D Numpy array of type int.
    :param depth: A int.

    :return: 2-D Numpy array of type int.
    """
    ret = np.zeros((x.shape[0], depth))
    ret[np.arange(x.shape[0]), x] = 1
    return ret

pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar

    if pbar is None:
        if total_size > 0:
            prefixes = ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi')
            power = min(int(math.log(total_size, 2) / 10), len(prefixes) - 1)
            scaled = float(total_size) / (2 ** (10 * power))
            total_size_str = '{:.1f} {}B'.format(scaled, prefixes[power])
            try:
                marker = '█'
            except UnicodeEncodeError:
                marker = '*'
            widgets = [
                progressbar.Percentage(),
                ' ', progressbar.DataSize(),
                ' / ', total_size_str,
                ' ', progressbar.Bar(marker=marker),
                ' ', progressbar.ETA(),
                ' ', progressbar.AdaptiveTransferSpeed(),
            ]
            pbar = progressbar.ProgressBar(widgets=widgets,
                                           max_value=total_size)
        else:
            widgets = [
                progressbar.DataSize(),
                ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
                ' ', progressbar.Timer(),
                ' ', progressbar.AdaptiveTransferSpeed(),
            ]
            pbar = progressbar.ProgressBar(widgets=widgets,
                                           max_value=progressbar.UnknownLength)

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path, show_progress)


def load_mnist_realval(path, one_hot=True, dequantify=False):
    """
    Loads the real valued MNIST dataset.

    :param path: Path to the dataset file.
    :param one_hot: Whether to use one-hot representation for the labels.
    :param dequantify:  Whether to add uniform noise to dequantify the data
        following (Uria, 2013).

    :return: The MNIST dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://www.iro.umontreal.ca/~lisa/deep/data/mnist'
                         '/mnist.pkl.gz', path)

    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    if dequantify:
        x_train += np.random.uniform(0, 1. / 256,
                                     size=x_train.shape).astype('float32')
        x_valid += np.random.uniform(0, 1. / 256,
                                     size=x_valid.shape).astype('float32')
        x_test += np.random.uniform(0, 1. / 256,
                                    size=x_test.shape).astype('float32')
    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return x_train, t_transform(t_train), x_valid, t_transform(t_valid), \
        x_test, t_transform(t_test)


def load_binary_mnist_realval(path):
    """
    Loads real valued MNIST dataset for binary classification (Treat 0 & 2-9
    as 0).

    :param path: Path to the dataset file.
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

    :param path: Path to the dataset file.
    :param one_hot: Whether to use one-hot representation for the labels.
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

    :param path: Path to the dataset file.
    :param normalize: Whether to normalize the x data to the range [0, 1].
    :param dequantify: Whether to add uniform noise to dequantify the data
        following (Uria, 2013).
    :param one_hot: Whether to use one-hot representation for the labels.

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
        with open(batch_file, 'rb') as f:
            if six.PY2:
                data = pickle.load(f)
            else:
                data = pickle.load(f, encoding='latin1')
            train_x.append(data['data'])
            train_y.append(data['labels'])
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)

    test_batch_file = os.path.join(batch_dir, 'test_batch')
    with open(test_batch_file, 'rb') as f:
        if six.PY2:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1')
        test_x = data['data']
        test_y = np.asarray(data['labels'])

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
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

    :param path: Path to the dataset file.
    :param normalize: Whether to normalize the x data to the range [0, 1].
    :param dequantify: Whether to add uniform noise to dequantify the data
        following (Uria, 2013).
    :param one_hot: Whether to use one-hot representation for the labels.
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

    x_train = data[:n_train, :n_dims]
    y_train = data[:n_train, n_dims] - 1
    x_test = data[n_train:, :n_dims]
    y_test = data[n_train:, n_dims] - 1

    return x_train, y_train, x_test, y_test


def load_uci_boston_housing(path, dtype=np.float32):
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://archive.ics.uci.edu/ml/'
                         'machine-learning-databases/housing/housing.data',
                         path)

    data = np.loadtxt(path)
    data = data.astype(dtype)
    permutation = np.random.choice(np.arange(data.shape[0]),
                                   data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.8))
    size_test = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0: size_train]
    index_test = permutation[size_train:size_test]
    index_val = permutation[size_test:]

    x_train, y_train = data[index_train, :-1], data[index_train, -1]
    x_val, y_val = data[index_val, :-1], data[index_val, -1]
    x_test, y_test = data[index_test, :-1], data[index_test, -1]

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_uci_protein_data(path, dtype=np.float32):
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://archive.ics.uci.edu/ml/'
                         'machine-learning-databases/00265/CASP.csv',
                         path)

    data = np.loadtxt(open(path), delimiter=',', skiprows=1)

    permutation = np.random.choice(np.arange(data.shape[0]),
                                   data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.8))
    size_test = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0: size_train]
    index_test = permutation[size_train:size_test]
    index_val = permutation[size_test:]

    x_train, y_train = data[index_train, 1:], data[index_train, 0]
    x_val, y_val = data[index_val, 1:], data[index_val, 0]
    x_test, y_test = data[index_test, 1:], data[index_test, 0]

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_uci_bow(data_name, data_path):
    """
    Loads the bag-of-words dataset from UCI machine learning repository.

    :param data_name: Name of the dataset, e.g., nips, NYTimes.
    :param data_path: Path to the dataset.

    :return: A tuple of (X, vocab), where X is a D * V bag-of-words matrix,
        whose each row is a document and its elements are count of each word.
        vocab is a list of words in the vocabulary.
    """
    data_dir = os.path.dirname(data_path)
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(data_dir)

    uci_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases' \
              '/bag-of-words/'
    vector_file = '{}.vector'.format(data_path)
    vocab_file = '{}.vocab'.format(data_path)
    numpy_file = '{}.npy'.format(data_path)

    if not os.path.isfile(numpy_file):
        download_dataset('{}docword.{}.txt.gz'.format(uci_url, data_name),
                         vector_file)
        with gzip.open(vector_file, 'rb') as f:
            D = int(f.readline())
            V = int(f.readline())
            T = int(f.readline())

            data = np.zeros((D, V), dtype=np.float32)
            for i in range(T):
                d, v, c = f.readline().split()
                data[int(d)-1, int(v)-1] += int(c)

        np.save(numpy_file, data)
        os.remove(vector_file)
    else:
        data = np.load(numpy_file)

    if not os.path.isfile(vocab_file):
        download_dataset('{}vocab.{}.txt'.format(uci_url, data_name),
                         vocab_file)

    with open(vocab_file) as vf:
        vocab = [v.strip() for v in vf.readlines()]

    return data, vocab


def load_uci_bow_sparse(data_name, data_path):
    """
    Loads the bag-of-words dataset from UCI machine learning repository.

    :param data_name: Name of the dataset, e.g., nips, NYTimes.
    :param data_path: Path to the dataset.

    :return: A tuple of (X, vocab), where X is a D * V bag-of-words matrix,
        whose each row is a document and its elements are count of each word.
        vocab is a list of words in the vocabulary.
    """
    data_dir = os.path.dirname(data_path)
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(data_dir)

    uci_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases' \
              '/bag-of-words/'
    vector_file = '{}.vector'.format(data_path)
    vocab_file = '{}.vocab'.format(data_path)

    if not os.path.isfile(vector_file):
        download_dataset('{}docword.{}.txt.gz'.format(uci_url, data_name),
                         vector_file)

    with gzip.open(vector_file, 'rb') as f:
        D = int(f.readline())
        V = int(f.readline())
        T = int(f.readline())
        data = [[] for _ in range(D)]

        for i in range(T):
            d, v, c = f.readline().split()
            data[int(d) - 1].append((int(v) - 1, int(c)))

    if not os.path.isfile(vocab_file):
        download_dataset('{}vocab.{}.txt'.format(uci_url, data_name),
                         vocab_file)

    with open(vocab_file) as vf:
        vocab = [v.strip() for v in vf.readlines()]

    return data, vocab


def load_movielens1m(path):
    """
    Loads the movielens 1M dataset.

    :param path: Path to the dataset.
    :return: The movielens 1M dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset(
            'http://files.grouplens.org/datasets/movielens/ml-1m.zip', path)

    zp = zipfile.ZipFile(path, 'r')
    content = zp.read('ml-1m/ratings.dat').decode('utf-8')
    data_list = content.split('\n')

    num_users = 0
    num_movies = 0
    corpus = []
    for item in data_list:
        term = item.split('::')
        if len(term) < 3:
            continue
        user_id = int(term[0]) - 1
        movie_id = int(term[1]) - 1
        rating = int(term[2])
        corpus.append((user_id, movie_id, rating))
        num_users = max(num_users, user_id + 1)
        num_movies = max(num_movies, movie_id + 1)

    corpus_data = np.array(corpus)
    np.random.shuffle(corpus_data)
    np.random.shuffle(corpus_data)
    N = np.shape(corpus_data)[0]
    Ndv = N // 20 * 17
    Ndv2 = N // 10 * 9
    train = corpus_data[:Ndv, :]
    valid = corpus_data[Ndv:Ndv2, :]
    test = corpus_data[Ndv2:, :]

    def find_non_trained(M, N, train_data):
        # Find non-trained files or peoples
        trained_movie = [False] * M
        trained_user = [False] * N
        for i in range(train_data.shape[0]):
            trained_user[train_data[i, 0]] = True
            trained_movie[train_data[i, 1]] = True
        us = 0
        vs = 0
        for i in range(N):
            us += trained_user[i]
        for j in range(M):
            vs += trained_movie[j]
        print('Untrained users = %d, untrained movies = %d'
              % (N - us, M - vs))

    find_non_trained(num_movies, num_users, train)
    return num_movies, num_users, train, valid, test


def load_movielens1m_mapped(path):
    num_movies, num_users, train, valid, test = load_movielens1m(path)

    user_movie = []
    user_movie_score = []
    for i in range(num_users):
        user_movie.append([])
        user_movie_score.append([])
    movie_user = []
    movie_user_score = []
    for i in range(num_movies):
        movie_user.append([])
        movie_user_score.append([])

    for i in range(np.shape(train)[0]):
        user_id = train[i, 0]
        movie_id = train[i, 1]
        rating = train[i, 2]
        user_movie[user_id].append(movie_id)
        user_movie_score[user_id].append(rating)
        movie_user[movie_id].append(user_id)
        movie_user_score[movie_id].append(rating)

    return num_movies, num_users, train, valid, test, \
        user_movie, user_movie_score, movie_user, movie_user_score
