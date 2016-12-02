#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.model import *
    from zhusuan.variational import advi, iwae
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class VariationalDropout():
    """
    A Bayesian neural network trained by variatioal dropout.


    :param layer_sizes: A list of Int. The dimensions of all layers.
    :param n: A Tensor or int. The number of data, or batch size in mini-batch
        training.
    :param n_particles: A Tensor or int. The number of particles per node.
    :param N: Int. The total number of training data.
    """
    def __init__(self, layer_sizes, n, n_particles, N):
        self.N = N
        self.n_class = layer_sizes[-1]

        with StochasticGraph() as model:
            logits_mu = tf.zeros([n_particles, n, self.n_class])
            logits_logstd = tf.zeros([n_particles, n, self.n_class])
            logits = Normal(logits_mu, logits_logstd)
            y = Discrete(logits.value)

        self.model = model
        self.n_particles = n_particles
        self.logits = logits
        self.y = y

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (n_particles, n, ...).
        :param observed: A dictionary of pairs: (string, Tensor). Each of
            the Tensor has shape (n, n_class).

        :return: A Tensor of shape (n_particles, n). The joint log likelihoods.
        """
        y = tf.tile(tf.expand_dims(observed['y'], 0), [self.n_particles, 1, 1])
        logits, y_out = self.model.get_output(
            [self.logits, self.y],
            inputs={self.logits: latent['logits'], self.y: y})

        cross_entropy = self.N * y_out[1]
        log_plogits = tf.reduce_sum(logits[1], 2)
        return cross_entropy + log_plogits

    def evaluate(self, logits, observed):
        """
        calculate the classification error rate.
        """
        logits = tf.reduce_mean(logits, 0)
        predict = tf.argmax(logits, 1)
        y = tf.argmax(observed['y'], 1)
        error = 1 - tf.reduce_mean(tf.cast(tf.equal(predict, y), tf.float32))
        return error


def q(x, layer_sizes, n_particles, is_training):
    """
    variational posterior.

    :param x: Tensor of shape (batch_size, input_size)
    :param layer_sizes: A list of Int. The dimensions of all layers.
    :param n_particles: A Tensor or int. The number of particles per node.
    :param is_training: Float32. 0 or 1.

    :return variational: StochasticGraph(). The variational graph.
    :return out: A Tensor of shape (n_particles, batch_size, n_class).
        Unnormalized log probability.
    """
    def _joinBiasDimension(input):
        """add bias 1"""
        return tf.concat(
            tf.rank(input)-1,
            [input, tf.expand_dims(tf.ones(tf.shape(input)[:-1]), -1)])

    def _my_batch_matmul(a, b):
        """
        param a: tensor of shape (n1,n2,n3)
        param b: tensor of shape (n3, n4)
        """
        c = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1, 1])
        return tf.batch_matmul(a, c)

    def _Real2Unit(input):
        """transforming real number to 0~1"""
        return 0.5 + tf.atan(input) / np.pi

    with StochasticGraph() as variational:
        denoising = is_training \
            * tf.random_normal(tf.shape(x), mean=1, stddev=0.5)\
            + 1. - is_training
        out = x * denoising
        out = tf.tile(tf.expand_dims(out, 0), [n_particles, 1, 1])

        for n_in, n_out, index in zip(layer_sizes[:-1], layer_sizes[1:],
                                      range(len(layer_sizes[:-1]))):
            W = tf.Variable(tf.random_normal([n_in+1, n_out]), name='W')
            alpha = tf.Variable(tf.zeros([n_in+1, n_out]), name='alpha')
            n_layer = tf.cast(n_in+1, tf.float32)

            out_mean = _my_batch_matmul(
                _joinBiasDimension(out) / n_layer**0.5, W)
            out_logstd = 0.5 * tf.log(_my_batch_matmul(
                _joinBiasDimension(out)**2 / n_layer,
                _Real2Unit(alpha) * W**2) + 1e-15)
            out = Normal(out_mean, out_logstd)
            if index < len(layer_sizes) - 2:
                out = tf.nn.relu(out.value)

    return variational, out


if __name__ == '__main__':
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.vstack([y_train, y_valid]).astype('float32')
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    n_x = x_train.shape[1]
    n_class = 10

    # Define training/evaluation parameters
    epoches = 500
    batch_size = 1000
    lb_samples = 1
    ll_samples = 20
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 3
    learning_rate = 0.01

    # Build trainging model
    learning_rate_ph = tf.placeholder(tf.float32, shape=())
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    is_training = tf.placeholder(tf.float32, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=(None, n_x))
    n = tf.shape(x)[0]
    layer_sizes = [n_x, 100, 100, 100, n_class]
    model = VariationalDropout(layer_sizes, n, n_particles, x_train.shape[0])
    y = tf.placeholder(tf.float32, shape=(None, n_class))
    observed = {'y': y}
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    variational, logits = q(x, layer_sizes, n_particles, is_training)
    logits = variational.get_output(logits)
    latent = {'logits': [logits[0], tf.reduce_sum(logits[1], 2)]}
    error = model.evaluate(latent['logits'][0], observed)

    lower_bound = tf.reduce_mean(advi(
        model, observed, latent, reduction_indices=0))
    infer = optimizer.minimize(-lower_bound)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer, lower_bound],
                    feed_dict={n_particles: lb_samples,
                               is_training: 1.0,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lb, test_error = sess.run(
                    [lower_bound, error],
                    feed_dict={n_particles: ll_samples,
                               is_training: 0.,
                               x: x_test, y: y_test})
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(test_lb))
                print('>> Test error = {}'.format(test_error))
