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
import utils


@zs.reuse('model')
def vae(observed, n, n_x, n_z, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        z_mean = tf.zeros([n, n_z])
        z_logstd = tf.zeros([n, n_z])
        z = zs.Normal('z', z_mean, z_logstd, n_samples=n_particles,
                      group_event_ndims=1)
        lx_z = layers.fully_connected(z, 500)
        lx_z = layers.fully_connected(lx_z, 500)
        x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)
        x = zs.Bernoulli('x', x_logits, group_event_ndims=1)
    return model


@zs.reuse('variational')
def q_net(observed, x, n_z, n_particles):
    with zs.BayesianNet(observed=observed) as variational:
        lz_x = layers.fully_connected(tf.to_float(x), 500)
        lz_x = layers.fully_connected(lz_x, 500)
        lz_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        lz_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = zs.Normal('z', lz_mean, lz_logstd, n_samples=n_particles,
                      group_event_ndims=1)
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
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 50
    ll_samples = 1000
    epoches = 3000
    batch_size = 1000
    iters = x_train.shape[0] // batch_size
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    test_n_temperatures = 100
    test_n_leapfrogs = 10
    test_n_chains = 10
    save_freq = 100
    result_path = "results/iwae/iwae50"

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    def log_joint(observed):
        model = vae(observed, n, n_x, n_z, n_particles)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    variational = q_net({}, x, n_z, n_particles)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    lower_bound = tf.reduce_mean(
        zs.iwae(log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz]}, axis=0))

    # Bidirectional Monte Carlo (BDMC) estimates of log likelihood:
    # Slower than IS estimates, used for evaluation after training
    def joint_obj(observed):
        return tf.squeeze(log_joint(observed))

    # Use q(z|x) as prior in BDMC
    def prior_obj(observed):
        z = observed['z']
        model = q_net({'z': z}, x, n_z, n_particles)
        log_qz = model.local_log_prob('z')
        return tf.squeeze(log_qz)

    prior_samples = {'z': qz_samples}
    z = tf.Variable(tf.zeros([1, test_n_chains * test_batch_size, n_z]),
                    name="z")
    hmc = zs.HMC(step_size=1e-6, n_leapfrogs=test_n_leapfrogs,
                 adapt_step_size=True, target_acceptance_rate=0.65,
                 adapt_mass=True)
    temperature = tf.placeholder(tf.float32, shape=[], name="temperature")
    bdmc = zs.BDMC(prior_obj, joint_obj, prior_samples, hmc,
                   {'x': x_obs}, {'z': z}, chain_axis=1,
                   n_chains=test_n_chains, n_temperatures=test_n_temperatures)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

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
                                            n_particles: lb_samples})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples})
                    test_lbs.append(test_lb)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test IWAE bound = {}'.format(np.mean(test_lbs)))

            if epoch % save_freq == 0:
                print('Saving model...')
                save_path = os.path.join(result_path,
                                         "iwae.epoch.{}.ckpt".format(epoch))
                utils.makedirs(save_path)
                saver.save(sess, save_path)
                print('Done')

        # BDMC evaluation
        print('Start evaluation...')
        time_bdmc = -time.time()
        test_ll_lbs = []
        test_ll_ubs = []
        for t in range(test_iters):
            test_x_batch = x_test[t * test_batch_size:
                                  (t + 1) * test_batch_size]
            test_x_batch = np.tile(test_x_batch, [test_n_chains, 1])
            ll_lb, ll_ub = bdmc.run(sess, feed_dict={x: test_x_batch,
                                                     n_particles: 1})
            test_ll_lbs.append(ll_lb)
            test_ll_ubs.append(ll_ub)
        time_bdmc += time.time()
        test_ll_lb = np.mean(test_ll_lbs)
        test_ll_ub = np.mean(test_ll_ubs)
        print('>> Test log likelihood (BDMC)\n>> lower bound = {}, '
              'upper bound = {}, BDMC gap = {}'
              .format(test_ll_lb, test_ll_ub, test_ll_ub - test_ll_lb))
