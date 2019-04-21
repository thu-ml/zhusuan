#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Check if the distribution-specific methods in BayesianNet process their
# arguments correctly.

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from zhusuan.framework import BayesianNet
from zhusuan.framework.utils import get_backward_ops


class TestNormal(tf.test.TestCase):
    def test_Normal(self):
        bn = BayesianNet()
        mean = tf.zeros([2, 3])
        logstd = tf.zeros([2, 3])
        std = tf.exp(logstd)
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.normal('a', mean, logstd=logstd, n_samples=n_samples,
                      group_ndims=group_ndims)
        b = bn.normal('b', mean, std=std, n_samples=n_samples,
                      group_ndims=group_ndims)

        for st in [a, b]:
            sample_ops = set(get_backward_ops(st.tensor))
            for i in [mean, logstd, n_samples]:
                self.assertTrue(i.op in sample_ops)
            log_p = st.log_prob(np.ones([2, 3]))
            log_p_ops = set(get_backward_ops(log_p))
            for i in [mean, logstd, group_ndims]:
                self.assertTrue(i.op in log_p_ops)
            self.assertEqual(a.get_shape()[1:], mean.get_shape())


class TestBernoulli(tf.test.TestCase):
    def test_Bernoulli(self):
        bn = BayesianNet()
        logits = tf.zeros([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.bernoulli('a', logits, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], logits.get_shape())


class TestCategorical(tf.test.TestCase):
    def test_Discrete(self):
        bn = BayesianNet()
        logits = tf.zeros([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=())
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.categorical('a', logits, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.array([0, 1]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], logits.get_shape()[:-1])


class TestUniform(tf.test.TestCase):
    def test_Uniform(self):
        bn = BayesianNet()
        minval = tf.zeros([2, 3])
        maxval = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.uniform('a', minval, maxval, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [minval, maxval, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.zeros([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [minval, maxval, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], minval.get_shape())


class TestGamma(tf.test.TestCase):
    def test_Gamma(self):
        bn = BayesianNet()
        alpha = tf.ones([2, 3])
        beta = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.gamma('a', alpha, beta, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [alpha, beta, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [alpha, beta, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], alpha.get_shape())


class TestBeta(tf.test.TestCase):
    def test_Beta(self):
        bn = BayesianNet()
        alpha = tf.ones([2, 3])
        beta = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.beta('a', alpha, beta, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [alpha, beta, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]) * 0.5)
        log_p_ops = set(get_backward_ops(log_p))
        for i in [alpha, beta, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], alpha.get_shape())


class TestPoisson(tf.test.TestCase):
    def test_Poisson(self):
        bn = BayesianNet()
        rate = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.poisson('a', rate, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [rate, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3], dtype=np.int32))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [rate, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], rate.get_shape())


class TestBinomial(tf.test.TestCase):
    def test_Binomial(self):
        bn = BayesianNet()
        logits = tf.zeros([2, 3])
        n_experiments = tf.placeholder(tf.int32, shape=[])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.binomial('a', logits, n_experiments, n_samples,
                        group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, n_experiments, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3], dtype=np.int32))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, n_experiments, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], logits.get_shape())


class TestMultinomial(tf.test.TestCase):
    def test_Multinomial(self):
        bn = BayesianNet()
        logits = tf.ones([2, 3])
        n_experiments = tf.placeholder(tf.int32, shape=[])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.multinomial('a', logits, n_experiments=n_experiments,
                           n_samples=n_samples,
                           group_ndims=group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, n_experiments, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3], dtype=np.int32))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, n_experiments, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], logits.get_shape())


class TestUnnormalizedMultinomial(tf.test.TestCase):
    def test_UnnormalizedMultinomial(self):
        bn = BayesianNet()
        logits = tf.ones([2, 3])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.unnormalized_multinomial('a', logits, group_ndims=group_ndims)
        log_p = a.log_prob(np.ones([2, 3], dtype=np.int32))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, group_ndims]:
            self.assertTrue(i.op in log_p_ops)


class TestOnehotCategorical(tf.test.TestCase):
    def test_OnehotCategorical(self):
        bn = BayesianNet()
        logits = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.onehot_categorical('a', logits, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(tf.one_hot([0, 2], 3, dtype=tf.int32))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], logits.get_shape())


class TestDirichlet(tf.test.TestCase):
    def test_Dirichlet(self):
        bn = BayesianNet()
        alpha = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.dirichlet('a', alpha, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [alpha, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [alpha, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], alpha.get_shape())


class TestInverseGamma(tf.test.TestCase):
    def test_InverseGamma(self):
        bn = BayesianNet()
        alpha = tf.ones([2, 3])
        beta = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.inverse_gamma('a', alpha, beta, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [alpha, beta, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [alpha, beta, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], alpha.get_shape())


class TestLaplace(tf.test.TestCase):
    def test_Laplace(self):
        bn = BayesianNet()
        loc = tf.zeros([2, 3])
        scale = tf.ones([2, 3])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.laplace('a', loc, scale, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [loc, scale, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [loc, scale, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], loc.get_shape())


class TestBinConcrete(tf.test.TestCase):
    def test_BinConcrete(self):
        bn = BayesianNet()
        logits = tf.zeros([2, 3])
        tau = tf.ones([])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.bin_concrete('a', tau, logits, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, tau, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, tau, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], logits.get_shape())


class TestConcrete(tf.test.TestCase):
    def test_Concrete(self):
        bn = BayesianNet()
        logits = tf.zeros([2, 3])
        tau = tf.ones([])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.concrete('a', tau, logits, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, tau, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, tau, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], logits.get_shape())


class TestExpConcrete(tf.test.TestCase):
    def test_ExpConcrete(self):
        bn = BayesianNet()
        logits = tf.zeros([2, 3])
        tau = tf.ones([])
        n_samples = tf.placeholder(tf.int32, shape=[])
        group_ndims = tf.placeholder(tf.int32, shape=[])
        a = bn.exp_concrete('a', tau, logits, n_samples, group_ndims)
        sample_ops = set(get_backward_ops(a.tensor))
        for i in [logits, tau, n_samples]:
            self.assertTrue(i.op in sample_ops)
        log_p = a.log_prob(np.ones([2, 3]))
        log_p_ops = set(get_backward_ops(log_p))
        for i in [logits, tau, group_ndims]:
            self.assertTrue(i.op in log_p_ops)
        self.assertEqual(a.get_shape()[1:], logits.get_shape())
