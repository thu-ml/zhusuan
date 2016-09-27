#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
import prettytensor as pt
from six.moves import range
import numpy as np
from dataset import standardize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.layers import *
    from zhusuan.variational import advi
    from zhusuan.evaluation import is_loglikelihood
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class Model():
    """
    A simple PBP model
    """
    def __init__(self, scale, N):
        self.scale = scale
        self.N = N

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.
        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (1, n_samples, n_latent).
        :param observed: A dictionary of pairs: (string, Tensor). Each of
            the Tenor has shape (batch_size, n_observed).

        :return: A Tensor of shape (batch_size, n_samples).
        The joint log likelihoods.
        """
        x = observed['x']
        y = observed['y']

        network_output = self.get_output_for(latent, x)
        network_output = tf.squeeze(network_output, [2, 3])
        y = tf.tile(tf.expand_dims(y, 1), [1, tf.shape(network_output)[1], 1])
        y = tf.squeeze(y, [2])
        log_likelihood = norm.logpdf(y, network_output, self.scale)
        log_likelihood = tf.reduce_mean(log_likelihood, 0) * self.N
        log_likelihood = tf.expand_dims(log_likelihood, 0)

        log_prior = sum([tf.reduce_sum(norm.logpdf(item))
                        for item in latent.values()])
        return log_likelihood + log_prior

    def get_output_for(self, latent, x):
        """
        get the network output of x with latent variables.
        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (1, n_samples, shape_latent).
        :param x: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_observed_x)

        :return: A Tensor of shape (batch_size, n_samples, n_observed_y)
        """
        w1 = latent['w1']
        w2 = latent['w2']

        w1 = tf.tile(w1, [tf.shape(x)[0], 1, 1, 1])
        w2 = tf.tile(w2, [tf.shape(x)[0], 1, 1, 1])

        x = tf.tile(tf.expand_dims(tf.expand_dims(x, 2), 1),
                    [1, tf.shape(w1)[1], 1, 1])
        x = tf.concat(2, [x, tf.ones((tf.shape(x)[0], tf.shape(x)[1], 1, 1))])

        l = tf.batch_matmul(w1, x) / \
            tf.sqrt(tf.cast(tf.shape(x)[2], tf.float32))
        l = tf.concat(2, [l, tf.ones((tf.shape(l)[0], tf.shape(l)[1], 1, 1))])
        l = tf.nn.relu(l)

        y = tf.batch_matmul(w2, l) / \
            tf.sqrt(tf.cast(tf.shape(l)[2], tf.float32))
        return y

    def rmse(self, latent, observed, std_y_train=1):
        """
        calculate the rmse.
        """
        network_output = self.get_output_for(latent, observed['x'])
        network_output = tf.squeeze(network_output, [3])
        network_output = tf.reduce_mean(network_output, 1)

        rmse = tf.sqrt(
            tf.reduce_mean((network_output
                            - observed['y'])**2)) * std_y_train
        return rmse


def get_data(name):
    data = np.loadtxt(name)

    # We obtain the features and the targets
    permutation = np.random.choice(range(data.shape[0]),
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


if __name__ == '__main__':
    tf.set_random_seed(1237)
    np.random.seed(1234)

    x_train, y_train, x_valid, y_valid, x_test, y_test \
        = get_data('data/2concrete.txt')
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.hstack((y_train, y_valid)).astype('float32')
    y_train = y_train.reshape((len(y_train), 1))

    x_test = x_test.astype('float32')
    y_test = y_test.reshape((len(y_test), 1))
    n_x = x_train.shape[1]

    x_train, x_test, _, _ = standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = standardize(y_train, y_test)

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 500
    epoches = 500
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 10
    learning_rate = 0.01
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75

    #build training model
    model = Model(0.1, x_train.shape[0])
    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    observed_x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
    observed_y = tf.placeholder(tf.float32, shape=(None, 1))
    observed = {'x': observed_x, 'y': observed_y}
    n_samples = tf.placeholder(tf.int32, shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    #shape is (batch_size, sample_num, shape_parameter)
    w1_mu = tf.Variable(tf.zeros([1, 1, 50, x_train.shape[1] + 1]))
    w1_logvar = tf.Variable(tf.zeros([1, 1, 50, x_train.shape[1] + 1]))
    w2_mu = tf.Variable(tf.zeros([1, 1, 1, 50 + 1]))
    w2_logvar = tf.Variable(tf.zeros([1, 1, 1, 50 + 1]))

    #build q
    w1_mu = InputLayer((1, 1, 50, x_train.shape[1] + 1), input=w1_mu)
    w1_logvar = InputLayer((1, 1, 50, x_train.shape[1] + 1), input=w1_logvar)
    w2_mu = InputLayer((1, 1, 1, 50 + 1), input=w2_mu)
    w2_logvar = InputLayer((1, 1, 1, 50 + 1), input=w2_logvar)
    w1 = ReparameterizedNormal([w1_mu, w1_logvar], n_samples)
    w2 = ReparameterizedNormal([w2_mu, w2_logvar], n_samples)
    latent = {'w1': get_output(w1), 'w2': get_output(w2)}

    lower_bound = tf.reduce_mean(advi(model,
                                 observed, latent, reduction_indices=1))
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)
    #infer = optimizer.minimize(-lower_bound)

    latent_outputs = {'w1': latent['w1'][0], 'w2': latent['w2'][0]}
    rmse = model.rmse(latent_outputs, observed, std_y_train)
    output = tf.reduce_mean(
        model.get_output_for(latent_outputs, observed_x), 1)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    init = tf.initialize_all_variables()

    # Run the inference
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]

                _, grad1, lb = sess.run(
                    [infer, grads, lower_bound],
                    feed_dict={n_samples: lb_samples,
                               learning_rate_ph: learning_rate,
                               observed_x: x_batch, observed_y: y_batch})

                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()

                test_lb, test_rmse = sess.run(
                    [lower_bound, rmse],
                    feed_dict={n_samples: ll_samples,
                               observed_x: x_test, observed_y: y_test})
                time_test += time.time()

                out1 = sess.run(output,
                                feed_dict={observed_x: x_test[0:6],
                                           n_samples: ll_samples})
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(test_lb))
                print('>> Test rmse = {}'.format(test_rmse))
                print(out1.reshape((1, 6)))
                print(y_test[0:10].transpose())
