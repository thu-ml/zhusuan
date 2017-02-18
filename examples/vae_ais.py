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
import zhusuan as zs

import dataset


@zs.reuse('model')
def vae(observed, n, n_x, n_z, n_particles, is_training):
    with zs.StochasticGraph(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_mean = tf.zeros([n_particles, n_z])
        z_logstd = tf.zeros([n_particles, n_z])
        z = zs.Normal('z', z_mean, z_logstd, sample_dim=1, n_samples=n)
        lx_z = layers.fully_connected(
            z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_z = layers.fully_connected(
            lx_z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)
        x = zs.Bernoulli('x', x_logits)
    return model


@zs.reuse('variational')
def q_net(observed, x, n_z, n_particles, is_training):
    with zs.StochasticGraph(observed=observed) as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        lz_x = layers.fully_connected(
            x, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.fully_connected(
            lz_x, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        lz_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = zs.Normal('z', lz_mean, lz_logstd, sample_dim=0,
                      n_samples=n_particles)
    return variational


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
    # x_test = x_test[:400, :]
    # t_test = t_test[:400]
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 1000
    epoches = 3000
    batch_size = 100
    test_batch_size = 400
    test_num_temperatures = 100
    test_num_leapfrogs = 10
    test_num_chains = 10
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 3000
    save_freq = 10
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    temperature = tf.placeholder(tf.float32, shape=[], name="temperature")
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    lz = tf.Variable(tf.zeros([1, test_num_chains * test_batch_size, n_z]), name="z")
    n = tf.shape(x)[0]

    def log_joint(observed):
        model = vae(observed, n, n_x, n_z, n_particles, is_training)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return tf.reduce_sum(log_pz, -1) + tf.reduce_sum(log_px_z, -1)

    variational = q_net({}, x, n_z, n_particles, is_training)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)

    log_qz = tf.reduce_sum(log_qz, -1)
    lower_bound = tf.reduce_mean(
        zs.advi(log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz]}, axis=0))
    log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x_obs},
                            {'z': [qz_samples, log_qz]}, axis=0))

    # For BDMC
    def joint_obj(observed):
        return tf.squeeze(log_joint(observed))

    # Use p(z) as prior
    # def prior_obj(observed):
    #     model = vae(observed, n, n_x, n_z, n_particles, is_training)
    #     log_pz = model.local_log_prob('z')
    #     return tf.squeeze(tf.reduce_sum(log_pz, -1))
    # vae_model = vae({}, n, n_x, n_z, n_particles, is_training)
    # prior_samples = {'z': vae_model.query('z', outputs=True)[0]}

    # Use q(z | x) as prior
    def prior_obj(observed):
        model = q_net(observed, observed['x'], n_z, n_particles, is_training)
        log_qz = model.local_log_prob('z')
        return tf.squeeze(tf.reduce_sum(log_qz, -1))

    prior_samples = {'z': qz_samples}

    hmc = zs.HMC(step_size=1e-6, n_leapfrogs=test_num_leapfrogs,
                 adapt_step_size=tf.constant(True), target_acceptance_rate=0.65,
                 adapt_mass=tf.constant(True))

    bdmc = zs.BDMC(prior_obj, joint_obj, prior_samples,
                   hmc, {'x': x_obs}, {'z': lz}, chain_axis=1,
                   num_chains=test_num_chains, num_temperatures=test_num_temperatures)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10, var_list=tf.all_variables())

    # Run the inference
    with tf.Session() as sess:
        def test():
            # IS test
            time_test = -time.time()
            test_lbs = []
            test_lls = []
            for t in range(test_iters):
                test_x_batch = x_test[
                               t * test_batch_size:(t + 1) * test_batch_size]
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

            # AIS test
            test_ll_lbs = []
            test_ll_ubs = []
            for t in range(test_iters):
                test_x_batch = x_test[
                               t * test_batch_size:(t + 1) * test_batch_size]
                test_x_batch = np.tile(test_x_batch, [test_num_chains, 1])

                ll_lb, ll_ub = bdmc.run(sess, feed_dict={x: test_x_batch,
                                                         n_particles: 1,
                                                         is_training: False})
                test_ll_lbs.append(ll_lb)
                test_ll_ubs.append(ll_ub)

            time_test += time.time()
            print('>>> TEST ({:.1f}s)'.format(time_test))
            print('>> Test VAE ELBO = {}, IWAE ELBO = {}'.format(np.mean(test_lbs), np.mean(test_lls)))
            test_ll_lb = np.mean(test_ll_lbs)
            test_ll_ub = np.mean(test_ll_ubs)
            print('>> Test log likelihood lower bound = {}, upper bound = {}, BDMC gap = {}'
                  .format(test_ll_lb, test_ll_ub, test_ll_ub - test_ll_lb))

        ckpt_file = tf.train.latest_checkpoint(".")

        # Restore
        begin_epoch = 1
        sess.run(tf.global_variables_initializer())
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)
            test()
        else:
            print('Starting from scratch...')

        for epoch in range(begin_epoch, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch_bin,
                                            learning_rate_ph: learning_rate,
                                            n_particles: lb_samples,
                                            is_training: True})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % save_freq == 0:
                print('Saving model...')
                save_path = "vae.epoch.{}.ckpt".format(epoch)
                saver.save(sess, save_path)
                print('Done')

            if epoch % test_freq == 0:
                test()