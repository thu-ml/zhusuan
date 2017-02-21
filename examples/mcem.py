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
    with zs.StochasticGraph(observed=observed) as model:
        z_mean = tf.zeros([n_particles, n_z])
        z_logstd = tf.zeros([n_particles, n_z])
        z = zs.Normal('z', z_mean, z_logstd, sample_dim=1, n_samples=n)
        lx_z = layers.fully_connected(z, 500)
        lx_z = layers.fully_connected(lx_z, 500)
        x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)
        x = zs.Bernoulli('x', x_logits)
    return model


@zs.reuse('variational')
def q_net(observed, x, n_z, n_particles):
    with zs.StochasticGraph(observed=observed) as variational:
        lz_x = layers.fully_connected(x, 500)
        lz_x = layers.fully_connected(lz_x, 500)
        lz_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        lz_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = zs.Normal('z', lz_mean, lz_logstd, sample_dim=0,
                      n_samples=n_particles, reparameterized=False)
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
    N, n_x = x_train.shape

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    rws_samples = 10
    ll_samples = 1000
    epoches = 3000
    batch_size = 1000
    iters = N // batch_size
    train_n_samples = 1
    train_n_leapfrogs = 10
    test_subset_size = 400
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    test_n_temperatures = 100
    test_n_leapfrogs = 10
    test_n_chains = 10
    test_freq = 10
    full_test_freq = 100
    save_freq = 100
    learning_rate = 0.003
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    result_path = "results/mcem"

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    temperature = tf.placeholder(tf.float32, shape=[], name="temperature")
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    last_z_train = np.zeros((x_train.shape[0], n_z))

    # ==== For optimize q ====
    def log_joint(observed):
        model = vae(observed, n, n_x, n_z, n_particles)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return tf.reduce_sum(log_pz, -1) + tf.reduce_sum(log_px_z, -1)

    variational = q_net({}, x, n_z, n_particles)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)

    log_qz = tf.reduce_sum(log_qz, -1)
    cost, ll_est = zs.rws(log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz]})
    cost = tf.reduce_mean(cost, axis=0)
    ll_est = tf.reduce_mean(ll_est, axis=0)

    log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x_obs},
                            {'z': [qz_samples, log_qz]}, axis=0))

    # ==== For BDMC ====
    def joint_obj(observed):
        return tf.squeeze(log_joint(observed))

    # Use q(z | x) as prior
    def prior_obj(observed):
        model = q_net(observed, observed['x'], n_z, n_particles)
        log_qz = model.local_log_prob('z')
        return tf.squeeze(tf.reduce_sum(log_qz, -1))

    prior_samples = {'z': qz_samples}
    eval_hmc = zs.HMC(step_size=1e-6, n_leapfrogs=test_n_leapfrogs,
                      adapt_step_size=True, target_acceptance_rate=0.65,
                      adapt_mass=True)
    eval_z = tf.Variable(tf.zeros([1, test_n_chains * test_batch_size, n_z]),
                         name="eval_z", trainable=False)
    bdmc = zs.BDMC(prior_obj, joint_obj, prior_samples,
                   eval_hmc, {'x': x_obs}, {'z': eval_z}, chain_axis=1,
                   n_chains=test_n_chains, n_temperatures=test_n_temperatures)

    # ==== For optimize p ====
    z = tf.Variable(tf.zeros([1, batch_size, n_z]), name="z", trainable=False)
    initialize_z = tf.assign(z, qz_samples)
    z_input = tf.placeholder(tf.float32, shape=z.get_shape())
    set_z = tf.assign(z, z_input)
    mcem_model = vae({'x': x_obs, 'z': z}, n, n_x, n_z, n_particles)
    log_px_z = mcem_model.local_log_prob('x')
    mcem_obj = tf.reduce_mean(tf.reduce_sum(log_px_z, -1))
    hmc = zs.HMC(step_size=1e-10, n_leapfrogs=train_n_leapfrogs,
                 adapt_step_size=True, target_acceptance_rate=0.65)
    sample_op = hmc.sample(joint_obj, {'x': x_obs}, {'z': z},
                           chain_axis=1)

    # ==== For MH step with q-net proposal ====
    mh_ratio = joint_obj({'x': x_obs, 'z': z}) - \
        prior_obj({'x': x_obs, 'z': z})
    log_score = joint_obj({'x': x_obs, 'z': z})

    # Gather variables
    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope="model")
    variational_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope="variational")
    print('Model parameters:')
    for i in model_params:
        print(i.name, i.get_shape())
    print('Variational parameters:')
    for i in variational_params:
        print(i.name, i.get_shape())

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    infer_model = optimizer.minimize(-mcem_obj, var_list=model_params)
    optimizer2 = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    infer_variational = optimizer2.minimize(cost,
                                            var_list=variational_params)

    saver = tf.train.Saver(max_to_keep=10)

    def bdmc_test(x_test):
        print('Start evaluation...')
        time_bdmc = -time.time()
        test_ll_lbs = []
        test_ll_ubs = []
        test_iters = x_test.shape[0] // test_batch_size
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
        print('>> Test log likelihood (BDMC) ({:.1f}s)\n'
              '>> lower bound = {}, upper bound = {}, BDMC gap = {}'
              .format(time_bdmc, test_ll_lb, test_ll_ub,
                      test_ll_ub - test_ll_lb))

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
            indices = np.random.permutation(N)
            x_train = x_train[indices, :]
            last_z_train = last_z_train[indices, :]

            lbs = []
            accs = []
            olds = []
            news = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
                z_batch = last_z_train[t * batch_size:(t + 1) * batch_size]

                # Compute old mh ratio
                sess.run(set_z, feed_dict={z_input:
                                           np.expand_dims(z_batch, 0)})
                old_mh_ratio, old_score = sess.run(
                    [mh_ratio, log_score],
                    feed_dict={x: x_batch_bin, n_particles: lb_samples})

                # Compute new mh ratio
                new_z_batch = np.squeeze(sess.run(
                    initialize_z, feed_dict={x: x_batch_bin,
                                             n_particles: lb_samples}))
                new_mh_ratio, new_score = sess.run(
                    [mh_ratio, log_score], {x: x_batch_bin,
                                            n_particles: lb_samples})

                olds.append(np.mean(old_score))
                news.append(np.mean(new_score))

                # Accept / reject
                acceptance_rate = np.exp(
                    np.minimum(0, new_mh_ratio - old_mh_ratio))
                mean_acc = np.mean(acceptance_rate)
                accs.append(mean_acc)
                if_accept = (np.random.rand(batch_size) <
                             acceptance_rate).astype(np.float32)
                if_accept = np.expand_dims(if_accept, 1)
                # if_accept = if_accept * 0
                # accepted_z_batch = if_accept * new_z_batch + \
                #     (1 - if_accept) * z_batch
                # accepted_z_batch = new_z_batch
                accepted_z_batch = z_batch

                # Store
                sess.run(set_z, feed_dict={
                    z_input: np.expand_dims(accepted_z_batch, 0)})

                # Sample z by HMC
                mcmc_z, _, _, _, oldp, newp, acc, ss = sess.run(
                    sample_op, feed_dict={x: x_batch_bin,
                                          n_particles: 1})

                # Store z back
                last_z_train[t * batch_size:(t + 1) * batch_size] = \
                    np.squeeze(mcmc_z)

                # Optimize p-net
                _ = sess.run(infer_model,
                             feed_dict={x: x_batch_bin,
                                        learning_rate_ph: learning_rate,
                                        n_particles: lb_samples})

                # Optimize q-net
                _, lb = sess.run([infer_variational, ll_est],
                                 feed_dict={x: x_batch_bin,
                                            learning_rate_ph: learning_rate,
                                            n_particles: rws_samples})
                lbs.append(lb)
            time_epoch += time.time()
            print(np.mean(olds), np.mean(news))
            print('Epoch {} ({:.1f}s): Lower bound = {}, '
                  'q acceptance rate = {}'.format(
                      epoch, time_epoch, np.mean(lbs), np.mean(accs)))

            if epoch % test_freq == 0:
                # IS test
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(ll_est,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: rws_samples})
                    test_ll = sess.run(log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))
                # BDMC test
                bdmc_test(x_test[:test_subset_size])

            if epoch % save_freq == 0:
                print('Saving model...')
                save_path = os.path.join(result_path,
                                         "mcem.epoch.{}.ckpt".format(epoch))
                utils.makedirs(save_path)
                saver.save(sess, save_path)
                print('Done')

        bdmc_test(x_test)
