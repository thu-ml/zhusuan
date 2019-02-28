#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from scipy import stats
import numpy as np
import six
import zhusuan as zs


def sample_error_with(sampler, sess, n_chains=1, n_iters=80000, thinning=50,
                      burnin=None, dtype=tf.float32):
    """
    Fig 1 in the SGHMC paper by Chen et al.
    """
    if burnin is None:
        burnin = n_iters * 2 // 3
    
    # Build the computation graph
    def log_joint(observed):
        x = observed['x']
        lh_noise = tf.random_normal(shape=tf.shape(x), stddev=2, dtype=dtype)
        return 2 * (x**2) - x**4 + lh_noise

    x = tf.Variable(
        tf.zeros([n_chains], dtype=dtype),
        trainable=False,
        name='x', dtype=dtype)
    sample_op, new_samples, sample_info = sampler.sample(
        log_joint, {}, {'x': x})

    # Run the inference
    sess.run(tf.global_variables_initializer())
    samples = []
    for t in range(n_iters):
        _, x_sample, info = sess.run(
            [sample_op, new_samples['x'], sample_info])
        if np.isnan(x_sample.sum()):
            raise ValueError('nan encountered')
        if t >= burnin and t % thinning == 0:
            samples.append(x_sample)
    samples = np.array(samples)
    samples = samples.reshape(-1)
    A = 3
    xs = np.linspace(-A, A, 1000)
    pdfs = np.exp(2*(xs**2) - xs**4)
    pdfs = pdfs / pdfs.mean() / A / 2
    est_pdfs = stats.gaussian_kde(samples)(xs)
    return np.abs(est_pdfs - pdfs).mean()


class TestSGMCMC(tf.test.TestCase):

    def test_sgld(self):
        sampler = zs.SGLD(learning_rate=0.01)
        with self.test_session() as sess:
            e = sample_error_with(sampler, sess, n_chains=100, n_iters=8000)
            # 6sd upper bound from 10 indepndent runs
            self.assertLessEqual(e, 0.023)

    def test_sghmc(self):
        sampler = zs.SGHMC(learning_rate=0.01, n_iter_resample_v=50,
                           friction=0.3, variance_estimate=0.02,
                           second_order=False)
        with self.test_session() as sess:
            e = sample_error_with(sampler, sess, n_chains=100, n_iters=8000)
            self.assertLessEqual(e, 0.016)

    def test_sghmc_second_order(self):
        sampler = zs.SGHMC(learning_rate=0.01, n_iter_resample_v=50,
                           friction=0.3, variance_estimate=0.02,
                           second_order=True)
        with self.test_session() as sess:
            e = sample_error_with(sampler, sess, n_chains=100, n_iters=8000)
            self.assertLessEqual(e, 0.016)

    def test_get_gradient_sum(self):
        n_train = 1000
        n_chains = 10
        x_dim = 100

        def log_joint(observed):
            x = observed['x']
            y = observed['y']
            w = observed['w']
            log_prior = tf.reduce_sum(w * w, -1)
            log_likelihood = (tf.einsum("ik,jk->ij", w, tf.convert_to_tensor(x)) - y) ** 2
            return log_prior + tf.reduce_mean(log_likelihood, -1) * n_train

        sampler = zs.SGLD(learning_rate=1e-3, add_noise=False)
        x = np.random.normal(size=[n_train, x_dim])
        w_truth = np.random.normal(size=[x_dim])
        y = np.matmul(x, w_truth) + np.random.normal(scale=x_dim*0.01, size=[n_train])
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        w_initial = np.random.normal(size=[n_chains, x_dim]).astype(np.float32)
        w = tf.Variable(w_initial, name='w')
        initialize_w = tf.assign(w, w_initial)
        # x = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')
        # y = tf.placeholder(tf.float32, shape=[None], name='y')
        sample_op1, new_samples1, _ = sampler.sample(
            log_joint, {'x': x, 'y': y}, {'w': w})
        sample_op2, new_samples2, _ = sampler.sample(
            log_joint, {'x': x, 'y': y}, {'w': w}, record_full=tf.convert_to_tensor(True), batch_size=100)
        sample_op3, new_samples3, _ = sampler.sample(
            log_joint, {'x': x, 'y': y}, {'w': w}, record_full=tf.convert_to_tensor(True), batch_size=89)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            _, w_sample1 = sess.run([sample_op1, new_samples1['w']])
            sess.run(initialize_w)
            _, w_sample2 = sess.run([sample_op2, new_samples2['w']])
            sess.run(initialize_w)
            _, w_sample3 = sess.run([sample_op3, new_samples3['w']])
            self.assertAllClose(w_sample1, w_sample2)
            self.assertAllClose(w_sample1, w_sample3)

