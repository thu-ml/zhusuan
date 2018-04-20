#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from six.moves import range
import numpy as np
import zhusuan as zs

from examples import conf
from examples.utils import dataset, save_image_collections


@zs.reuse('model')
def vae(observed, x_dim, z_dim, n, n_particles=1):
    with zs.BayesianNet(observed=observed) as model:
        z_mean = tf.zeros([n, z_dim])
        z = zs.Normal('z', z_mean, std=1., group_ndims=1, n_samples=n_particles)
        lx_z = tf.layers.dense(z, 500, activation=tf.nn.relu)
        lx_z = tf.layers.dense(lx_z, 500, activation=tf.nn.relu)
        x_logits = tf.layers.dense(lx_z, x_dim)
        x_mean = zs.Implicit("x_mean", tf.sigmoid(x_logits), group_ndims=1)
        x = zs.Bernoulli('x', x_logits, group_ndims=1)
    return model


def q_net(observed, x_dim, z_dim, n_z_per_x):
    with zs.BayesianNet(observed=observed) as variational:
        x = zs.Empirical('x', tf.int32, (None, x_dim))
        lz_x = tf.layers.dense(tf.to_float(x), 500, activation=tf.nn.relu)
        lz_x = tf.layers.dense(lz_x, 500, activation=tf.nn.relu)
        z_mean = tf.layers.dense(lz_x, z_dim)
        z_logstd = tf.layers.dense(lz_x, z_dim)
        z = zs.Normal('z', z_mean, logstd=z_logstd, group_ndims=1,
                      n_samples=n_z_per_x)
    return variational


def main():
    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    x_dim = x_train.shape[1]

    # Define model parameters
    z_dim = 40

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')
    x = tf.to_int32(tf.less(tf.random_uniform(tf.shape(x_input)), x_input))
    n = tf.shape(x)[0]

    def log_joint(observed):
        model = vae(observed, x_dim, z_dim, n, n_particles)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    variational = q_net({'x': x}, x_dim, z_dim, n_particles)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    lower_bound = zs.variational.elbo(log_joint,
                                      observed={'x': x},
                                      latent={'z': [qz_samples, log_qz]},
                                      axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    # Importance sampling estimates of marginal log likelihood
    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x},
                            {'z': [qz_samples, log_qz]}, axis=0))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)

    # Generate images
    n_gen = 100
    x_mean = vae({}, x_dim, z_dim, n_gen).outputs('x_mean')
    x_gen = tf.reshape(x_mean, [-1, 28, 28, 1])

    # Define training/evaluation parameters
    epochs = 3000
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    save_freq = 10
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    result_path = "results/vae"

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            n_particles: 1})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: 1})
                    test_ll = sess.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: 1000})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))

            if epoch % save_freq == 0:
                images = sess.run(x_gen)
                name = os.path.join(result_path,
                                    "vae.epoch.{}.png".format(epoch))
                save_image_collections(images, name)


if __name__ == "__main__":
    main()
