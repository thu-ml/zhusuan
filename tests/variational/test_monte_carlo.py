#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from scipy import stats
import numpy as np
import tensorflow as tf

from zhusuan.variational.monte_carlo import *
from zhusuan.distributions import Normal
from tests.variational.utils import _kl_normal_normal


class TestImportanceWeightedObjective(tf.test.TestCase):
    def setUp(self):
        self._rng = np.random.RandomState(1)
        self._n1_samples = self._rng.standard_normal(size=(1, 1000)).\
            astype(np.float32)
        self._n3_samples = self._rng.standard_normal(1000).astype(np.float32)
        super(TestImportanceWeightedObjective, self).setUp()

    def test_objective(self):
        log_qx_n1 = stats.norm.logpdf(self._n1_samples).astype(np.float32)
        qx_samples_n1 = tf.convert_to_tensor(self._n1_samples)
        log_qx_n1 = tf.convert_to_tensor(log_qx_n1)

        log_qx_n3 = stats.norm.logpdf(self._n3_samples).astype(np.float32)
        qx_samples_n3 = tf.convert_to_tensor(self._n3_samples)
        log_qx_n3 = tf.convert_to_tensor(log_qx_n3)

        def _check_k1_elbo(x_mean, x_std):
            # check their elbo
            def log_joint(observed):
                norm = Normal(mean=x_mean, std=x_std)
                return norm.log_prob(observed['x'])

            lower_bound = importance_weighted_objective(
                log_joint, observed={},
                latent={'x': [qx_samples_n1, log_qx_n1]}, axis=0)
            lower_bound = tf.reduce_mean(lower_bound)
            analytic_lower_bound = -_kl_normal_normal(0., 1., x_mean, x_std)
            with self.session(use_gpu=True) as sess:
                a = sess.run(lower_bound)
                b = sess.run(analytic_lower_bound)
                self.assertNear(a, b, 1e-2)

        _check_k1_elbo(0., 1.)
        _check_k1_elbo(2., 3.)

        def _check_monotonous_elbo(x_mean, x_std):
            # check their elbo
            def log_joint(observed):
                norm = Normal(mean=x_mean, std=x_std)
                return norm.log_prob(observed['x'])

            lower_bound = importance_weighted_objective(
                log_joint, observed={},
                latent={'x': [qx_samples_n3, log_qx_n3]}, axis=0)
            lower_bound = tf.reduce_mean(lower_bound)
            analytic_lower_bound = -_kl_normal_normal(0., 1., x_mean, x_std)
            with self.session(use_gpu=True) as sess:
                a = sess.run(lower_bound)
                b = sess.run(analytic_lower_bound)
                self.assertTrue(a > b - 1e-6)

        _check_monotonous_elbo(0., 1.)
        _check_monotonous_elbo(2., 3.)

    def test_sgvb(self):
        eps_samples = tf.convert_to_tensor(self._n1_samples)
        mu = tf.constant(2.)
        sigma = tf.constant(3.)
        qx_samples = eps_samples * sigma + mu
        norm = Normal(mean=mu, std=sigma)
        log_qx = norm.log_prob(qx_samples)

        def _check_sgvb(x_mean, x_std, threshold):
            def log_joint(observed):
                norm = Normal(mean=x_mean, std=x_std)
                return norm.log_prob(observed['x'])

            lower_bound = importance_weighted_objective(
                log_joint, observed={}, latent={'x': [qx_samples, log_qx]},
                axis=0)
            sgvb_cost = lower_bound.sgvb()
            sgvb_cost = tf.reduce_mean(sgvb_cost)
            sgvb_grads = tf.gradients(sgvb_cost, [mu, sigma])
            true_cost = _kl_normal_normal(mu, sigma, x_mean, x_std)
            true_grads = tf.gradients(true_cost, [mu, sigma])

            with self.session(use_gpu=True) as sess:
                g1 = sess.run(sgvb_grads)
                g2 = sess.run(true_grads)
                # print('sgvb_grads:', g1)
                # print('true_grads:', g2)
                self.assertAllClose(g1, g2, threshold, threshold)

        _check_sgvb(0., 1., 0.04)
        _check_sgvb(2., 3., 0.02)

    def test_vimco(self):
        eps_samples = tf.convert_to_tensor(self._n3_samples)
        mu = tf.constant(2.)
        sigma = tf.constant(3.)
        qx_samples = eps_samples * sigma + mu
        norm = Normal(mean=mu, std=sigma)
        log_qx = norm.log_prob(qx_samples)

        v_qx_samples = eps_samples * tf.stop_gradient(sigma) + \
            tf.stop_gradient(mu)
        v_log_qx = norm.log_prob(v_qx_samples)

        def _check_vimco(x_mean, x_std, threshold):
            def log_joint(observed):
                norm = Normal(mean=x_mean, std=x_std)
                return norm.log_prob(observed['x'])

            lower_bound = importance_weighted_objective(
                log_joint, observed={}, latent={'x': [qx_samples, log_qx]},
                axis=0)
            v_lower_bound = importance_weighted_objective(
                log_joint, observed={}, latent={'x': [v_qx_samples, v_log_qx]},
                axis=0)

            vimco_cost = v_lower_bound.vimco()
            vimco_cost = tf.reduce_mean(vimco_cost)
            vimco_grads = tf.gradients(vimco_cost, [mu, sigma])
            sgvb_cost = tf.reduce_mean(lower_bound.sgvb())
            sgvb_grads = tf.gradients(sgvb_cost, [mu, sigma])

            with self.session(use_gpu=True) as sess:
                g1 = sess.run(vimco_grads)
                g2 = sess.run(sgvb_grads)
                # print('vimco_grads:', g1)
                # print('sgvb_grads:', g2)
                self.assertAllClose(g1, g2, threshold, threshold)

        _check_vimco(0., 1., 1e-2)
        _check_vimco(2., 3., 1e-6)
