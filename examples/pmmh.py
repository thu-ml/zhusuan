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
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs

import dataset
import utils


@zs.reuse('generator')
def generator(observed, n, n_z, is_training):
    with zs.StochasticGraph(observed=observed) as generator:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        ngf = 64
        # z_min = -tf.ones([n_z])
        # z_max = tf.ones([n_z])
        # z = zs.Uniform('z', z_min, z_max, sample_dim=0, n_samples=n)
        z_mean = tf.zeros([n_z])
        z_logstd = tf.zeros([n_z])
        z = zs.Normal('z', z_mean, z_logstd, sample_dim=0, n_samples=n)
        lx_z = layers.fully_connected(z, num_outputs=ngf*8*4*4,
                                      normalizer_fn=layers.batch_norm,
                                      normalizer_params=normalizer_params)
        lx_z = tf.reshape(lx_z, [-1, 4, 4, ngf*8])
        lx_z = layers.conv2d_transpose(lx_z, ngf*4, 5, stride=2,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(lx_z, ngf*2, 5, stride=2,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(lx_z, 3, 5, stride=2,
                                       activation_fn=tf.nn.sigmoid)
    return generator, lx_z


def pmmh(sess, x_generator, z_ph, x, n_z, is_training,
         jumps=200, burn_in=100, epsilon=0.1, proposal_sigma=1.):
    n = x.shape[0]
    z = np.random.randn(n, n_z)
    x_gen = sess.run(x_generator, feed_dict={z_ph: z, is_training: False})
    log_p = np.sum(
        stats.norm.logpdf(x, loc=x_gen, scale=np.ones_like(x) * epsilon),
            axis=(1, 2, 3)) + np.sum(stats.norm.logpdf(z), axis=-1)

    trace = [z]
    samples = []
    jump_sigma = np.ones_like(z) * proposal_sigma
    acc_cnts = np.zeros([n], dtype='float')
    for jump in range(jumps):
        z_next = stats.norm.rvs(loc=z, scale=jump_sigma)
        log_p_jump = np.sum(
            stats.norm.logpdf(z_next, loc=z, scale=jump_sigma), axis=-1)
        x_gen_next = sess.run(x_generator, feed_dict={z_ph: z_next,
                                                      is_training: False})
        log_p_next = np.sum(
            stats.norm.logpdf(x, loc=x_gen_next,
                              scale=np.ones_like(x) * epsilon),
            axis=(1, 2, 3)) + np.sum(stats.norm.logpdf(z_next), axis=-1)
        log_p_back = np.sum(stats.norm.logpdf(z, loc=z_next, scale=jump_sigma),
                            axis=-1)
        acc_rate = np.exp(log_p_next + log_p_back - log_p - log_p_jump)
        print("MH jump: {}".format(jump))
        condition = np.random.uniform(size=acc_rate.shape) <= acc_rate
        acc_cnts += condition
        z = np.where(np.expand_dims(condition, 1), z_next, z)
        log_p = np.where(condition, log_p_next, log_p)
        x_gen = np.where(np.expand_dims(condition, 1),
                         x_gen_next.reshape(n, -1),
                         x_gen.reshape(n, -1)).reshape(x.shape)
        err = np.mean((x_gen - x) ** 2)
        print("Err: {}".format(err))
        if jump >= burn_in:
            samples.append(x_gen.reshape(x.shape))
        trace.append(z)
    print("Acceptance rates:")
    print(acc_cnts / jumps)
    return samples, trace


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load CIFAR
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'cifar10', 'cifar-10-python.tar.gz')
    np.random.seed(1234)
    x_train, t_train, x_test, t_test = \
        dataset.load_cifar10(data_path, normalize=True, one_hot=True)
    _, n_xl, _, n_channels = x_train.shape
    # x_train -= np.mean(x_train, 0)
    # x_test -= np.mean(x_train, 0)
    n_y = t_train.shape[1]

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    batch_size = 64
    iters = 1000
    anneal_lr_freq = 1e37
    anneal_lr_rate = 0.75
    learning_rate = 0.1
    test_batch = x_test[:batch_size]

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=(None, n_xl, n_xl, n_channels),
                       name='x')

    # sgd for inference of z
    z_sgd = tf.Variable(tf.zeros([batch_size, n_z]))
    model, x_sgd = generator({'z': z_sgd}, batch_size, n_z, is_training)
    loss = tf.reduce_mean(tf.square(x - x_sgd))
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    sgd_infer = optimizer.minimize(loss, var_list=[z_sgd])

    # pmmh for inference of z
    z_ph = tf.placeholder(tf.float32, shape=[batch_size, n_z])
    _, x_generator = generator({'z': z_ph}, batch_size, n_z, is_training)

    # Create saver to load DCGAN model
    gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='generator')
    saver = tf.train.Saver(var_list=gen_var_list)

    for i in gen_var_list:
        print(i.name, i.get_shape())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        save_path = "results/dcgan/dcgan.epoch.{}.iter.{}.ckpt". \
            format(199, 1562)
        saver.restore(sess, save_path)

        # sgd on z
        # for iter in range(1, iters + 1):
        #     if iter % anneal_lr_freq == 0:
        #         learning_rate *= anneal_lr_rate
        #     _, loss_, x_sgd_ = sess.run([sgd_infer, loss, x_sgd],
        #                                 feed_dict={x: test_batch,
        #                                 learning_rate_ph: learning_rate,
        #                                 is_training: False})
        #     print('Iter={}: loss = {}'.format(iter, loss_))


        x_samples, trace = pmmh(
            sess, x_generator, z_ph, test_batch, n_z, is_training,
            jumps=100000, burn_in=50000, epsilon=0.01, proposal_sigma=0.05)

        orig = "results/pmmh/test.png"
        utils.save_image_collections(test_batch, orig, shape=(8, 8),
                                     scale_each=True)
        # name = "results/pmmh/x_sgd.png"
        # utils.save_image_collections(x_sgd_, name, shape=(8, 8),
        #                              scale_each=True)
        for i, x_sample in enumerate(x_samples[-100:]):
            name = "results/pmmh/x_samples/x_sample.{}.png".format(i)
            utils.save_image_collections(x_sample, name, shape=(8, 8),
                                         scale_each=True)
