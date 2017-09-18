#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from scipy import stats
import numpy as np
import tensorflow as tf

from zhusuan.variational.exclusive_kl import *
from zhusuan.distributions import Normal

from tests.variational.utils import _kl_normal_normal


class TestEvidenceLowerBound(tf.test.TestCase):
    def setUp(self):
        # print a list of samples from normal
        self._rng = np.random.RandomState(1)
        self._n01_samples = self._rng.standard_normal(1000).astype(np.float32)
        super(TestEvidenceLowerBound, self).setUp()

    def test_objective(self):
        log_qx = stats.norm.logpdf(self._n01_samples).astype(np.float32)
        qx_samples = tf.convert_to_tensor(self._n01_samples)
        log_qx = tf.convert_to_tensor(log_qx)

        def _check_elbo(x_mean, x_std):
            # check their elbo
            def log_joint(observed):
                norm = Normal(mean=x_mean, std=x_std)
                return norm.log_prob(observed['x'])

            lower_bound = elbo(log_joint, observed={},
                               latent={'x': [qx_samples, log_qx]}, axis=0)
            analytic_lower_bound = -_kl_normal_normal(0., 1., x_mean, x_std)
            with self.test_session(use_gpu=True) as sess:
                a = sess.run(lower_bound)
                b = sess.run(analytic_lower_bound)
                # print(a, b)
                self.assertNear(a, b, 1e-2)

        _check_elbo(0., 1.)
        _check_elbo(2., 3.)

    def test_sgvb(self):
        eps_samples = tf.convert_to_tensor(self._n01_samples)
        mu = tf.constant(2.)
        sigma = tf.constant(3.)
        qx_samples = eps_samples * sigma + mu
        norm = Normal(mean=mu, std=sigma)
        log_qx = norm.log_prob(qx_samples)

        def _check_sgvb(x_mean, x_std, threshold):
            def log_joint(observed):
                norm = Normal(mean=x_mean, std=x_std)
                return norm.log_prob(observed['x'])

            lower_bound = elbo(log_joint, observed={},
                               latent={'x': [qx_samples, log_qx]}, axis=0)
            sgvb_cost = lower_bound.sgvb()
            sgvb_grads = tf.gradients(sgvb_cost, [mu, sigma])
            true_cost = _kl_normal_normal(mu, sigma, x_mean, x_std)
            true_grads = tf.gradients(true_cost, [mu, sigma])

            with self.test_session(use_gpu=True) as sess:
                g1 = sess.run(sgvb_grads)
                g2 = sess.run(true_grads)
                # print('sgvb_grads:', g1)
                # print('true_grads:', g2)
                self.assertAllClose(g1, g2, threshold, threshold)

        _check_sgvb(0., 1., 0.04)
        # 1e-6 would be good for sgvb if sticking the landing is used. (p=q)
        _check_sgvb(2., 3., 0.02)

    def test_reinforce(self):
        eps_samples = tf.convert_to_tensor(self._n01_samples)
        mu = tf.constant(2.)
        sigma = tf.constant(3.)
        qx_samples = tf.stop_gradient(eps_samples * sigma + mu)
        norm = Normal(mean=mu, std=sigma)
        log_qx = norm.log_prob(qx_samples)

        def _check_reinforce(x_mean, x_std, threshold):
            def log_joint(observed):
                norm = Normal(mean=x_mean, std=x_std)
                return norm.log_prob(observed['x'])

            lower_bound = elbo(log_joint, observed={},
                               latent={'x': [qx_samples, log_qx]}, axis=0)
            # TODO: Check grads when use variance reduction and baseline
            reinforce_cost = lower_bound.reinforce(variance_reduction=False)
            reinforce_grads = tf.gradients(reinforce_cost, [mu, sigma])
            true_cost = _kl_normal_normal(mu, sigma, x_mean, x_std)
            true_grads = tf.gradients(true_cost, [mu, sigma])

            with self.test_session(use_gpu=True) as sess:
                sess.run(tf.global_variables_initializer())
                g1 = sess.run(reinforce_grads)
                g2 = sess.run(true_grads)
                # print('reinforce_grads:', g1)
                # print('true_grads:', g2)
                self.assertAllClose(g1, g2, threshold, threshold)

        _check_reinforce(0., 1., 0.03)
        # asymptotically no variance (p=q)
        _check_reinforce(2., 3., 1e-6)
