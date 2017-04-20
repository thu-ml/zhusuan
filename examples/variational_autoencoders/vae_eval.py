#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation of trained variational autoencoders using Bidirectional Monte
Carlo (Grosse, 2015), (Wu, 2016).
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from six.moves import range
import numpy as np
import zhusuan as zs
from zhusuan.evaluation import BDMC

from examples import conf
from examples.utils import dataset
from examples.variational_autoencoders.vae import vae, q_net


if __name__ == "__main__":
    tf.set_random_seed(1237)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 40

    # Define evaluation parameters
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    test_n_temperatures = 100
    test_n_leapfrogs = 10
    test_n_chains = 10
    result_path = "results/vae"

    # Build the computation graph
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [test_n_chains, 1, 1])
    n = tf.shape(x)[0]

    def log_joint(observed):
        model = vae(observed, n, n_x, n_z, test_n_chains, False)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    # Bidirectional Monte Carlo (BDMC) estimates of log likelihood:
    # Slower than IS estimates, used for offline evaluation.

    # Use q(z|x) as prior in BDMC
    def log_qz_given_x(observed):
        z = observed['z']
        model = q_net({'z': z}, x, n_z, test_n_chains, False)
        return model.local_log_prob('z')

    variational = q_net({}, x, n_z, test_n_chains, False)
    prior_samples = {'z': variational.outputs('z')}
    z = tf.Variable(tf.zeros([test_n_chains, test_batch_size, n_z]),
                    name="z", trainable=False)
    hmc = zs.HMC(step_size=1e-6, n_leapfrogs=test_n_leapfrogs,
                 adapt_step_size=True, target_acceptance_rate=0.65,
                 adapt_mass=True)
    bdmc = BDMC(log_qz_given_x, log_joint, prior_samples, hmc,
                {'x': x_obs}, {'z': z},
                n_chains=test_n_chains, n_temperatures=test_n_temperatures)

    model_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope="model")
    variational_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope="variational")
    saver = tf.train.Saver(max_to_keep=10,
                           var_list=model_var_list + variational_var_list)

    # Run the evaluation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            saver.restore(sess, ckpt_file)

        # BDMC evaluation
        print('Start evaluation...')
        time_bdmc = -time.time()
        test_ll_lbs = []
        test_ll_ubs = []
        for t in range(test_iters):
            time_iter = -time.time()
            test_x_batch = x_test[t * test_batch_size:
                                  (t + 1) * test_batch_size]
            ll_lb, ll_ub = bdmc.run(sess, feed_dict={x: test_x_batch})
            time_iter += time.time()
            print('Test batch {} ({:.1f}s): lower bound = {}, upper bound = {}'
                  .format(t, time_iter, ll_lb, ll_ub))
            test_ll_lbs.append(ll_lb)
            test_ll_ubs.append(ll_ub)
        time_bdmc += time.time()
        test_ll_lb = np.mean(test_ll_lbs)
        test_ll_ub = np.mean(test_ll_ubs)
        print('>> Test log likelihood (BDMC) ({:.1f}s)\n'
              '>> lower bound = {}, upper bound = {}, BDMC gap = {}'
              .format(time_bdmc, test_ll_lb, test_ll_ub,
                      test_ll_ub - test_ll_lb))
