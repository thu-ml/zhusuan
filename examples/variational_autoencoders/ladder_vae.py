#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ladder Variational Autoencoder on CIFAR10. (Casper, 2016)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from collections import namedtuple

import numpy as np
import six
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import layers
import zhusuan as zs
from zhusuan.distributions_old import logistic

from examples import conf
from examples.utils import dataset, multi_gpu, optimizers
from examples.utils.multi_gpu import FLAGS


@zs.reuse('model')
def ladder_vae(observed, n, n_particles, groups):
    with zs.BayesianNet(observed=observed) as model:
        h_top = tf.get_variable(name='h_top_p',
                                shape=[1, 1, 1, groups[-1].num_filters],
                                initializer=tf.constant_initializer(0.0))
        h = downward(h_top, n, n_particles, groups)
        h = tf.nn.elu(h)
        x = layers.conv2d_transpose(h, 3, kernel_size=5, stride=2,
                                    activation_fn=None, scope='x_mean')
        x = tf.reshape(x, [n_particles, n, -1])
        x_logsd = tf.get_variable("x_logsd", (),
                                  initializer=tf.constant_initializer(0.0))
    return model, x, x_logsd


@zs.reuse('variational')
def q_net(x, n_xl, n_particles, groups):
    with zs.BayesianNet() as variational:
        x = tf.reshape(x, [-1, n_xl, n_xl, 3])
        n = tf.shape(x)[0]
        h = layers.conv2d(x, groups[0].num_filters, 5, 2, activation_fn=None)
        h, qz_mean, qz_logstd = upward(h, n, n_particles, groups)
        h_top = tf.get_variable(name='h_top_q',
                                shape=[1, 1, 1, groups[-1].num_filters],
                                initializer=tf.constant_initializer(0.0))
        h = downward(h_top, n, n_particles, groups,
                     lateral_inputs=[qz_mean, qz_logstd])
    z_names = list(six.iterkeys(qz_mean))
    return variational, z_names


def upward(h, n, n_particles, groups):
    qz_mean = {}
    qz_logstd = {}
    for group_i, group in enumerate(groups):
        for block_i in range(group.num_blocks):
            name = 'group_%d/block_%d' % (group_i, block_i)
            stride = 1
            if group_i > 0 and block_i == 0:
                stride = 2
            h1 = tf.nn.elu(h)
            h1 = layers.conv2d(h1, group.num_filters + 2 * group.n_z, 3,
                               stride=stride, activation_fn=None,
                               scope=name + '_up_conv1')
            qz_mean[name], qz_logstd[name], h1 = tf.split(
                h1, [group.n_z] * 2 + [group.num_filters], axis=3)
            h1 = tf.nn.elu(h1)
            h1 = layers.conv2d(h1, group.num_filters, kernel_size=3,
                               activation_fn=None, scope=name + '_up_conv2')
            if stride == 2:
                h = layers.conv2d(h, group.num_filters, 3, stride=2,
                                  activation_fn=None,
                                  scope=name + '_resize_up')
            h += 0.1 * h1
    return h, qz_mean, qz_logstd


def downward(h_top, n, n_particles, groups, lateral_inputs=None):
    h = tf.tile(h_top, [n * n_particles, groups[-1].map_size,
                        groups[-1].map_size, 1])
    for group_i, group in reversed(list(enumerate(groups))):
        for block_i in reversed(range(group.num_blocks)):
            name = 'group_%d/block_%d' % (group_i, block_i)
            stride = 1
            if group_i > 0 and block_i == 0:
                stride = 2
            h1 = tf.nn.elu(h)
            h1 = layers.conv2d_transpose(
                h1, group.n_z * 2 + group.num_filters, 3,
                activation_fn=None, scope=name + '_down_conv1')
            h1, pz_mean, pz_logstd = tf.split(
                h1, [group.num_filters] + [group.n_z] * 2, axis=3)

            if lateral_inputs:
                qz_mean, qz_logstd = lateral_inputs
                pz_mean, pz_logstd = [
                    tf.reshape(x, [n_particles, -1, group.map_size,
                                   group.map_size, group.n_z])
                    for x in [pz_mean, pz_logstd]]
                post_z_mean = pz_mean + tf.expand_dims(qz_mean[name], 0)
                post_z_logstd = pz_logstd + tf.expand_dims(qz_logstd[name], 0)
                post_z_mean, post_z_logstd = [
                    tf.reshape(x, [-1, group.map_size, group.map_size,
                                   group.n_z])
                    for x in [post_z_mean, post_z_logstd]]
                z_i = zs.Normal(name, post_z_mean, post_z_logstd,
                                group_event_ndims=3)
            else:
                z_i = zs.Normal(name, pz_mean, pz_logstd,
                                group_event_ndims=3)

            z_i = tf.reshape(z_i, [-1, group.map_size, group.map_size,
                                   group.n_z])
            h1 = tf.concat([h1, z_i], 3)
            h1 = tf.nn.elu(h1)
            if stride == 2:
                h = layers.conv2d_transpose(h, group.num_filters, 3, 2,
                                            scope=name + '_resize_down')
            h1 = layers.conv2d_transpose(h1, group.num_filters, 3,
                                         stride=stride,
                                         activation_fn=None,
                                         scope=name + '_down_conv2')
            h += 0.1 * h1
    return h


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load CIFAR
    data_path = os.path.join(conf.data_dir, 'cifar10',
                             'cifar-10-python.tar.gz')
    np.random.seed(1234)
    x_train, t_train, x_test, t_test = \
        dataset.load_cifar10(data_path, normalize=True, one_hot=True)
    _, n_xl, _, n_channels = x_train.shape
    n_x = n_xl * n_xl * n_channels
    x_train = x_train.reshape((-1, n_x))
    x_train -= np.mean(x_train, 0)
    x_test = x_test.reshape((-1, n_x))
    x_test -= np.mean(x_train, 0)
    n_y = t_train.shape[1]

    # Define training/evaluation parameters
    lb_samples = 1
    epoches = 1000
    batch_size = 32 * FLAGS.num_gpus
    test_batch_size = 32 * FLAGS.num_gpus
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    print_freq = 100
    test_freq = iters
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    bottle_neck_group = namedtuple(
        'bottle_neck_group',
        ['num_blocks', 'num_filters', 'map_size', 'n_z'])
    groups = [
        bottle_neck_group(2, 64, 16, 64),
        bottle_neck_group(2, 64, 8, 64),
        bottle_neck_group(2, 64, 4, 64)
    ]

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=(None, n_x), name='x')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = optimizers.AdamaxOptimizer(learning_rate_ph, beta1=0.9,
                                           beta2=0.999)

    def build_tower_graph(x, id_):
        x_part = x[id_ * tf.shape(x)[0] // FLAGS.num_gpus:
                   (id_ + 1) * tf.shape(x)[0] // FLAGS.num_gpus]
        n = tf.shape(x_part)[0]
        x_obs = tf.tile(tf.expand_dims(x_part, 0), [n_particles, 1, 1])

        def log_joint(observed):
            obs_ = observed.copy()
            x = obs_.pop('x')
            model, x_mean, x_logsd = ladder_vae(obs_, n, n_particles, groups)
            log_pz = model.local_log_prob(list(six.iterkeys(obs_)))
            log_px_z = tf.log(logistic.cdf(x + 1. / 256, x_mean, x_logsd) -
                              logistic.cdf(x, x_mean, x_logsd) + 1e-8)
            log_px_z = tf.reduce_sum(log_px_z, -1)
            return log_px_z + tf.add_n(log_pz)

        variational, z_names = q_net(x_part, n_xl, n_particles, groups)
        qz_outputs = variational.query(z_names, outputs=True,
                                       local_log_prob=True)
        latents = dict(zip(z_names, qz_outputs))
        lower_bound = tf.reduce_mean(
            zs.sgvb(log_joint, {'x': x_obs}, latents, axis=0))
        bits_per_dim = -lower_bound / n_x * 1. / np.log(2.)
        grads = optimizer.compute_gradients(bits_per_dim)
        return lower_bound, bits_per_dim, grads

    tower_losses = []
    tower_grads = []
    for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
                lower_bound, bits_per_dim, grads = build_tower_graph(x, i)
                tower_losses.append([lower_bound, bits_per_dim])
                tower_grads.append(grads)
    lower_bound, bits_per_dim = multi_gpu.average_losses(tower_losses)
    grads = multi_gpu.average_gradients(tower_grads)
    infer = optimizer.apply_gradients(grads)

    total_size = 0
    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())
        total_size += np.prod([int(s) for s in i.get_shape()])
    print("Num trainable variables: %d" % total_size)

    # Run the inference
    with multi_gpu.create_session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs, test_lbs, test_lls = [], [], []
            bits, test_bits = [], []
            time_train = -time.time()
            for t in range(iters):
                iter = t + 1
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                try:
                    _, lb, bit = sess.run(
                        [infer, lower_bound, bits_per_dim],
                        feed_dict={x: x_batch,
                                   learning_rate_ph: learning_rate,
                                   n_particles: lb_samples})
                except tf.errors.InvalidArgumentError as error:
                    if ("NaN" in error.message) or ("Inf" in error.message):
                        continue
                    raise
                lbs.append(lb)
                bits.append(bit)

                if iter % print_freq == 0:
                    print('Epoch={} Iter={} ({:.3f}s/iter): '
                          'Lower bound = {} bits = {}'.
                          format(epoch, iter,
                                 (time.time() + time_train) / print_freq,
                                 np.mean(lbs), np.mean(bits)))
                    lbs = []
                    bits = []

                if iter % test_freq == 0:
                    time_test = -time.time()
                    test_lbs = []
                    test_bits = []
                    for tt in range(test_iters):
                        test_x_batch = x_test[tt * test_batch_size:
                                              (tt + 1) * test_batch_size]
                        test_lb, test_bit = sess.run(
                            [lower_bound, bits_per_dim],
                            feed_dict={x: test_x_batch,
                                       n_particles: lb_samples})
                        test_lbs.append(test_lb)
                        test_bits.append(test_bit)
                    time_test += time.time()
                    print('>>> TEST ({:.1f}s)'.format(time_test))
                    print('>> Test lower bound = {} bits = {}'.format(
                        np.mean(test_lbs), np.mean(test_bits)))

                if iter % print_freq == 0:
                    time_train = -time.time()
