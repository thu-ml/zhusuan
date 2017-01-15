#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.model import *
    from zhusuan.variational import advi
    from zhusuan.evaluation import is_loglikelihood
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_z: A Tensor or int. The dimension of latent variables (z).
    :param n_x: A Tensor or int. The dimension of observed variables (x).
    :param n: A Tensor or int. The number of data, or batch size in mini-batch
        training.
    :param n_particles: A Tensor or int. The number of particles per node.
    """
    def __init__(self, n_z, n_x, n, n_particles, is_training):
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        with StochasticGraph() as model:
            z_mean = tf.zeros([n_particles, n_z])
            z_logstd = tf.zeros([n_particles, n_z])
            z = Normal(z_mean, z_logstd, sample_dim=1, n_samples=n)
            lx_z = tf.reshape(z.value, [-1, 1, 1, n_z])
            lx_z = layers.conv2d_transpose(
                lx_z, 128, kernel_size=3, padding='VALID',
                normalizer_fn=layers.batch_norm,
                normalizer_params=normalizer_params
            )
            lx_z = layers.conv2d_transpose(
                lx_z, 64, kernel_size=5, padding='VALID',
                normalizer_fn=layers.batch_norm,
                normalizer_params=normalizer_params
            )
            lx_z = layers.conv2d_transpose(
                lx_z, 32, kernel_size=5, stride=2,
                normalizer_fn=layers.batch_norm,
                normalizer_params=normalizer_params
            )
            lx_z = layers.conv2d_transpose(
                lx_z, 1, kernel_size=5, stride=2,
                activation_fn=None,
            )
            lx_z = tf.reshape(lx_z, [n_particles, n, -1])
            x = Bernoulli(lx_z)
        self.model = model
        self.x = x
        self.z = z
        self.n_particles = n_particles

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor).
        :param observed: A dictionary of pairs: (string, Tensor).
        :param given: A dictionary of pairs: (string, Tensor).

        :return: A Tensor. The joint log likelihoods.
        """
        z = latent['z']
        x = observed['x']
        x = tf.tile(tf.expand_dims(x, 0), [self.n_particles, 1, 1])
        z_out, x_out = self.model.get_output([self.z, self.x],
                                             inputs={self.z: z, self.x: x})
        log_px_z = tf.reduce_sum(x_out[1], -1)
        log_pz = tf.reduce_sum(z_out[1], -1)
        return log_px_z + log_pz


def q_net(x, n_xl, n_z, n_particles, is_training):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param x: A Tensor.
    :param n_xl: A Tensor or int. The dimension of observed variable (x) width.
    :param n_z: A Tensor or int. The dimension of latent variables (z).
    :param n_particles: A Tensor or int. Number of samples of latent variables.
    """
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}
    with StochasticGraph() as variational:
        lz_x = tf.reshape(x, [-1, n_xl, n_xl, 1])
        lz_x = layers.conv2d(
            lz_x, 32, kernel_size=5, stride=2,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.conv2d(
            lz_x, 64, kernel_size=5, stride=2,
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.conv2d(
            lz_x, 128, kernel_size=5, padding='VALID',
            normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.dropout(lz_x, keep_prob=0.9, is_training=is_training)
        lz_x = tf.reshape(lz_x, [-1, 128*3*3])
        lz_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        lz_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = Normal(lz_mean, lz_logstd, sample_dim=0, n_samples=n_particles)
    return variational, z


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]
    n_xl = int(np.sqrt(n_x))

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 100
    epoches = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    n = tf.shape(x)[0]
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    model = M1(n_z, n_x, n, n_particles, is_training)
    variational, lz = q_net(x, n_xl, n_z, n_particles, is_training)
    z, z_logpdf = variational.get_output(lz)
    z_logpdf = tf.reduce_sum(z_logpdf, -1)
    lower_bound = tf.reduce_mean(advi(
        model, {'x': x}, {'z': [z, z_logpdf]}, reduction_indices=0))
    log_likelihood = tf.reduce_mean(is_loglikelihood(
        model, {'x': x}, {'z': [z, z_logpdf]}, reduction_indices=0))

    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
    #                                       tf.get_default_graph())
    # train_writer.close()

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch = np.random.binomial(
                    n=1, p=x_batch, size=x_batch.shape).astype('float32')
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch,
                                            learning_rate_ph: learning_rate,
                                            n_particles: lb_samples,
                                            is_training: True})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))
            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[
                        t * test_batch_size: (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples,
                                                  is_training: False})
                    test_ll = sess.run(log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  is_training: False})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))
