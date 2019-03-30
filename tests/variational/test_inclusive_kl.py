#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import warnings

from scipy import stats
import numpy as np
import tensorflow as tf

from zhusuan.variational.inclusive_kl import *
from zhusuan.distributions import Normal

from tests.variational.utils import _kl_normal_normal


class TestInclusiveKLObjective(tf.test.TestCase):
    def setUp(self):
        # print a list of samples from normal
        self._rng = np.random.RandomState(1)
        self._n01_samples = self._rng.standard_normal(1000).astype(np.float32)
        super(TestInclusiveKLObjective, self).setUp()

    def test_objective(self):
        log_qx = stats.norm.logpdf(self._n01_samples).astype(np.float32)
        qx_samples = tf.convert_to_tensor(self._n01_samples)
        log_qx = tf.convert_to_tensor(log_qx)

        def log_joint(observed):
            norm = Normal(std=1.)
            return norm.log_prob(observed['x'])

        lower_bound = klpq(log_joint, observed={},
                           latent={'x': [qx_samples, log_qx]}, axis=0)
        err_msg = "can only be optimized instead of being evaluated"
        with self.assertRaisesRegexp(NotImplementedError, err_msg):
            _ = lower_bound + 1.
        with self.session(use_gpu=True) as sess:
            with self.assertRaisesRegexp(NotImplementedError, err_msg):
                sess.run(lower_bound)

    def test_rws(self):
        eps_samples = tf.convert_to_tensor(self._n01_samples)
        mu = tf.constant(2.)
        sigma = tf.constant(3.)
        qx_samples = tf.stop_gradient(eps_samples * sigma + mu)
        q = Normal(mean=mu, std=sigma)
        log_qx = q.log_prob(qx_samples)

        def _check_rws(x_mean, x_std, threshold):
            def log_joint(observed):
                p = Normal(mean=x_mean, std=x_std)
                return p.log_prob(observed['x'])

            klpq_obj = klpq(log_joint, observed={},
                            latent={'x': [qx_samples, log_qx]}, axis=0)
            cost = klpq_obj.rws()
            rws_grads = tf.gradients(cost, [mu, sigma])
            true_cost = _kl_normal_normal(x_mean, x_std, mu, sigma)
            true_grads = tf.gradients(true_cost, [mu, sigma])

            with self.session(use_gpu=True) as sess:
                g1 = sess.run(rws_grads)
                g2 = sess.run(true_grads)
                # print('rws_grads:', g1)
                # print('true_grads:', g2)
                self.assertAllClose(g1, g2, threshold, threshold)

        _check_rws(0., 1., 0.01)
        _check_rws(2., 3., 0.02)

        single_sample = tf.stop_gradient(tf.random_normal([]) * sigma + mu)
        single_log_q = q.log_prob(single_sample)

        def log_joint(observed):
            p = Normal(std=1.)
            return p.log_prob(observed['x'])

        single_sample_obj = klpq(
            log_joint, observed={},
            latent={'x': [single_sample, single_log_q]})

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            single_sample_obj.rws()
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue("biased and inaccurate when you're using only "
                            "a single sample" in str(w[-1].message))
