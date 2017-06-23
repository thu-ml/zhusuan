#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset, save_image_collections


def main():
    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.random.binomial(1, x_train, size=x_train.shape)
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 40

    @zs.reuse('model')
    def vae(observed, n, n_x, n_z):
        with zs.BayesianNet(observed=observed) as model:
            z_mean = tf.zeros([n, n_z])
            z = zs.Normal('z', z_mean, std=1., group_event_ndims=1)
            lx_z = layers.fully_connected(z, 500)
            lx_z = layers.fully_connected(lx_z, 500)
            x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)
            x = zs.Bernoulli('x', x_logits, group_event_ndims=1)
        return model, x_logits

    @zs.reuse('variational')
    def q_net(x, n_z):
        with zs.BayesianNet() as variational:
            lz_x = layers.fully_connected(tf.to_float(x), 500)
            lz_x = layers.fully_connected(lz_x, 500)
            z_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
            z_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
            z = zs.Normal('z', z_mean, logstd=z_logstd, group_event_ndims=1)
        return variational

    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    n = tf.shape(x)[0]

    def log_joint(observed):
        model, _ = vae(observed, n, n_x, n_z)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    variational = q_net(x, n_z)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint,
                observed={'x': x},
                latent={'z': [qz_samples, log_qz]}))

    optimizer = tf.train.AdamOptimizer(0.001)
    infer = optimizer.minimize(-lower_bound)

    # Generate images
    n_gen = 100
    _, x_logits = vae({}, n_gen, n_x, n_z)
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])

    # Define training parameters
    epochs = 500
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    save_freq = 1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch})
                lbs.append(lb)

            print('Epoch {}: Lower bound = {}'.format(
                epoch, np.mean(lbs)))

            if epoch % save_freq == 0:
                images = sess.run(x_gen)
                name = "results/vae/vae.epoch.{}.png".format(epoch)
                save_image_collections(images, name)


if __name__ == "__main__":
    main()
