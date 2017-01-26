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
    from zhusuan.variational import vimco
    from zhusuan.evaluation import is_loglikelihood
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()

class M3:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_z: A Tensor or int. The dimension of latent variables (z).
    :param n_x: A Tensor or int. The dimension of observed variables (x).
    :param n: A Tensor or int. The number of data, or batch size in mini-batch
        training.
    :param n_particles: A Tensor or int. The number of particles per node.
    """
    def __init__(self, n_z, n_x, n, n_particles, is_training):
        with StochasticGraph() as model:
            z_mean = tf.zeros([n_particles, n_z])
            z = Bernoulli(z_mean, sample_dim=1, n_samples=n)
            lh_z = layers.fully_connected(z.value, 200, activation_fn=None)
            h2 = Bernoulli(lh_z)

            lh_h = layers.fully_connected(h2.value, 200, activation_fn=None)
            h1 = Bernoulli(lh_h)

            lx_h = layers.fully_connected(h1.value, n_x, activation_fn=None)
            x = Bernoulli(lx_h)

        self.model = model
        self.x = x
        self.h1 = h1
        self.h2 = h2
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
        h1 = latent['h1']
        h2 = latent['h2']
        x = observed['x']
        x = tf.tile(tf.expand_dims(x, 0), [self.n_particles, 1, 1])
        z_out, x_out, h1_out, h2_out = self.model.get_output([self.z, self.x, self.h1, self.h2],
                                             inputs={self.z: z, self.x: x, self.h1: h1, self.h2: h2})
        log_px_h1 = tf.reduce_sum(x_out[1], -1)
        log_ph1_h2 = tf.reduce_sum(h1_out[1], -1)
        log_ph2_z = tf.reduce_sum(h2_out[1], -1)
        log_pz = tf.reduce_sum(z_out[1], -1)
        return log_px_h1 + log_ph1_h2 + log_ph2_z + log_pz


def q_net(x, n_z, n_particles, is_training):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param x: A Tensor.
    :param n_x: A Tensor or int. The dimension of observed variables (x).
    :param n_z: A Tensor or int. The dimension of latent variables (z).
    :param n_particles: A Tensor or int. Number of samples of latent variables.
    """
    with StochasticGraph() as variational:
        lh_x = layers.fully_connected(x, 200, activation_fn=None)
        h1 = Bernoulli(lh_x, sample_dim=0, n_samples=n_particles)

        lh_h = layers.fully_connected(h1.value, 200, activation_fn=None)
        h2 = Bernoulli(lh_h)

        lz_h = layers.fully_connected(h2.value, n_z, activation_fn=None)
        z = Bernoulli(lz_h)

    return variational, z, h2, h1

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
    n_z = 200

    # Define training/evaluation parameters
    lb_samples = 2
    ll_samples = 500
    epoches = 3000
    batch_size = 24
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
    model = M3(n_z, n_x, n, n_particles, is_training)
    variational, lz, lh2, lh1 = q_net(x, n_z, n_particles, is_training)
    z, z_logpdf = variational.get_output(lz)
    h1, h1_logpdf = variational.get_output(lh1)
    h2, h2_logpdf = variational.get_output(lh2)

    z_logpdf = tf.reduce_sum(z_logpdf, -1)
    h1_logpdf = tf.reduce_sum(h1_logpdf, -1)
    h2_logpdf = tf.reduce_sum(h2_logpdf, -1)

    object_function, lower_bound = vimco(
        model, {'x': x}, {'z': [z, z_logpdf], 'h1': [h1, h1_logpdf],
                          'h2': [h2, h2_logpdf]}, reduction_indices=0, is_particle_larger_one=True)

    lower_bound = tf.reduce_mean(lower_bound)
    object_function = tf.reduce_mean(object_function)

    log_likelihood = tf.reduce_mean(is_loglikelihood(
        model, {'x': x}, {'z': [z, z_logpdf], 'h1': [h1, h1_logpdf],
                          'h2': [h2, h2_logpdf]}, reduction_indices=0))

    grads = optimizer.compute_gradients(-object_function)
    infer = optimizer.apply_gradients(grads)

    # train_writer = tf.train.SummaryWriter('/tmp/zhusuan',
    #                                       tf.get_default_graph())
    # train_writer.close()

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    # logTrainFile = open('log_K_2_train.csv', 'w')
    # logTestFile = open('log_K_2_test.csv', 'w')
    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
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
                _, lb, of = sess.run([infer, lower_bound, object_function],
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
                # logTestFile.write('%lf,%lf\n' % (np.mean(test_lb), np.mean(test_ll)))
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood = {}'.format(np.mean(test_lls)))

    # logTestFile.close()
    # logTrainFile.close()