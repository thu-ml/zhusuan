#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from zhusuan.model.stochastic import *
from zhusuan.model.base import BayesianNet
from zhusuan.model.utils import get_backward_ops


class TestNormal(tf.test.TestCase):
    def test_Normal(self):
        with BayesianNet():
            mean = tf.zeros([2, 3])
            logstd = tf.zeros([2, 3])
            std = tf.exp(logstd)
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Normal('a', mean, logstd=logstd, n_samples=n_samples,
                       group_event_ndims=group_event_ndims)
            b = Normal('b', mean, std=std, n_samples=n_samples,
                       group_event_ndims=group_event_ndims)

        for st in [a, b]:
            sample_ops = set(get_backward_ops(st.tensor))
            for i in [mean, logstd, n_samples]:
                self.assertTrue(i.op in sample_ops)
            log_p = st.log_prob(np.ones([2, 3]))
            log_p_ops = set(get_backward_ops(log_p))
            for i in [mean, logstd, group_event_ndims]:
                self.assertTrue(i.op in log_p_ops)


class TestBernoulli(tf.test.TestCase):
    def test_Bernoulli(self):
        with BayesianNet():
            logits = tf.zeros([2, 3])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Bernoulli('a', logits, n_samples, group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestCategorical(tf.test.TestCase):
    def test_Discrete(self):
        with BayesianNet():
            logits = tf.zeros([2, 3])
            n_samples = tf.placeholder(tf.int32, shape=())
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Categorical('a', logits, n_samples, group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.array([0, 1]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestUniform(tf.test.TestCase):
    def test_Uniform(self):
        with BayesianNet():
            minval = tf.zeros([2, 3])
            maxval = tf.ones([2, 3])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Uniform('a', minval, maxval, n_samples, group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [minval, maxval, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.zeros([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [minval, maxval, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestGamma(tf.test.TestCase):
    def test_Gamma(self):
        with BayesianNet():
            alpha = tf.ones([2, 3])
            beta = tf.ones([2, 3])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Gamma('a', alpha, beta, n_samples, group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [alpha, beta, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [alpha, beta, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestBeta(tf.test.TestCase):
    def test_Beta(self):
        with BayesianNet():
            alpha = tf.ones([2, 3])
            beta = tf.ones([2, 3])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Beta('a', alpha, beta, n_samples, group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [alpha, beta, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]) * 0.5)
        log_p_ops = set(get_backward_ops(log_p))
        for i in [alpha, beta, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestPoisson(tf.test.TestCase):
    def test_Poisson(self):
        with BayesianNet():
            rate = tf.ones([2, 3])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Poisson('a', rate, n_samples, group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [rate, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3], dtype=np.int32))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [rate, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestBinomial(tf.test.TestCase):
    def test_Binomial(self):
        with BayesianNet():
            logits = tf.zeros([2, 3])
            n_experiments = tf.placeholder(tf.int32, shape=[])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Binomial('a', logits, n_experiments, n_samples,
                         group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, n_experiments, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3], dtype=np.int32))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, n_experiments, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestMultinomial(tf.test.TestCase):
    def test_Multinomial(self):
        with BayesianNet():
            logits = tf.ones([2, 3])
            n_experiments = tf.placeholder(tf.int32, shape=[])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Multinomial('a', logits, n_experiments, n_samples,
                            group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, n_experiments, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3], dtype=np.int32))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, n_experiments, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestOnehotCategorical(tf.test.TestCase):
    def test_OnehotCategorical(self):
        with BayesianNet():
            logits = tf.ones([2, 3])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = OnehotCategorical('a', logits, n_samples, group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(tf.one_hot([0, 2], 3, dtype=tf.int32))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestDirichlet(tf.test.TestCase):
    def test_Dirichlet(self):
        with BayesianNet():
            alpha = tf.ones([2, 3])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Dirichlet('a', alpha, n_samples, group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [alpha, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [alpha, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestInverseGamma(tf.test.TestCase):
    def test_InverseGamma(self):
        with BayesianNet():
            alpha = tf.ones([2, 3])
            beta = tf.ones([2, 3])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = InverseGamma('a', alpha, beta, n_samples, group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [alpha, beta, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [alpha, beta, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestLaplace(tf.test.TestCase):
    def test_Laplace(self):
        with BayesianNet():
            loc = tf.zeros([2, 3])
            scale = tf.ones([2, 3])
            n_samples = tf.placeholder(tf.int32, shape=[])
            group_event_ndims = tf.placeholder(tf.int32, shape=[])
            a = Laplace('a', loc, scale, n_samples, group_event_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [loc, scale, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [loc, scale, group_event_ndims]:
            self.assertTrue(i.op in log_p_ops)
