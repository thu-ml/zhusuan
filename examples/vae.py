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

    :param n_z: Int. The dimension of latent variables (z).
    :param n_x: Int. The dimension of observed variables (x).
    """
    def __init__(self, n_z, n_x, n, n_particles):
        self.n_z = n_z
        self.n_x = n_x
        with StochasticGraph() as model:
            z_mean = tf.zeros([n_particles, n_z])
            z_logstd = tf.zeros([n_particles, n_z])
            z = Normal(z_mean, z_logstd, sample_dim=1, n_samples=n)
            lx_z = layers.fully_connected(z.value, 500)
            lx_z = layers.fully_connected(lx_z, 500)
            lx_z = layers.fully_connected(lx_z, n_x)
            x = Bernoulli(lx_z)
        self.model = model
        self.x = x
        self.z = z
        self.n_particles = n_particles

    def log_prob(self, latent, observed, given):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_samples, n_latent).
        :param observed: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_observed).

        :return: A Tensor of shape (batch_size, n_samples). The joint log
            likelihoods.
        """
        z = latent['z']
        x = observed['x']
        x = tf.tile(tf.expand_dims(x, 0), [self.n_particles, 1, 1])
        z_out, x_out = self.model.get_output([self.z, self.x],
                                             inputs={self.z: z, self.x: x})

        log_px_z = tf.reduce_sum(x_out[1], -1)
        log_pz = tf.reduce_sum(z_out[1], -1)

        return log_px_z + log_pz


def q_net(n_x, n_z, n_particles):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param n_x: Int. The dimension of observed variables (x).
    :param n_z: Int. The dimension of latent variables (z).
    :param n_samples: A Int or a Tensor of type int. Number of samples of
        latent variables.

    :return: All :class:`Layer` instances needed.
    """
    with StochasticGraph() as variational:
        x = tf.placeholder(tf.float32, shape=(None, n_x))
        lz_x = layers.fully_connected(x, 500)
        lz_x = layers.fully_connected(lz_x, 500)
        lz_mean = layers.fully_connected(lz_x, n_z)
        lz_logstd = layers.fully_connected(lz_x, n_z)
        z = Normal(lz_mean, lz_logstd, sample_dim=0, n_samples=n_particles)
    return variational, x, z


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

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 5000
    epoches = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    n = tf.placeholder(tf.int32, shape=[])
    n_particles = tf.placeholder(tf.int32, shape=[])

    def build_model(reuse=False):
        with tf.variable_scope("model", reuse=reuse) as scope:
            model = M1(n_z, x_train.shape[1], n, n_particles)
        with tf.variable_scope("variational", reuse=reuse) as scope:
            variational, lx, lz = q_net(n_x, n_z, n_particles)
        return model, variational, lx, lz

    # Build the training computation graph
    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    model, variational, lx, lz = build_model(reuse=False)
    z_outputs = variational.get_output(lz, {lx: x})
    lower_bound = tf.reduce_mean(advi(
        model, {'x': x}, {'z': z_outputs}, reduction_indices=0))
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    # Build the evaluation computation graph
    eval_model, eval_variational, eval_lx, eval_lz = build_model(reuse=True)
    z_outputs = eval_variational.get_output(eval_lz, {eval_lx: x})
    eval_lower_bound = tf.reduce_mean(advi(
        eval_model, {'x': x}, {'z': z_outputs}, reduction_indices=0))
    eval_log_likelihood = tf.reduce_mean(is_loglikelihood(
        eval_model, {'x': x}, {'z': z_outputs}, reduction_indices=0))

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
                                            n: batch_size})
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
                    test_lb = sess.run(eval_lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples,
                                                  n: test_batch_size})
                    test_ll = sess.run(eval_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  n: test_batch_size})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))
