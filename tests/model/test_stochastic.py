#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from .context import zhusuan
from zhusuan.model.stochastic import *
from zhusuan.model.base import StochasticGraph
from zhusuan.model.utils import get_backward_tensors


class TestNormal:
    def test_init(self):
        with StochasticGraph() as model:
            mean = tf.zeros([2, 3])
            logstd = tf.zeros([2, 3])
            sample_dim = tf.placeholder(tf.int32, shape=())
            n_samples = tf.placeholder(tf.int32, shape=())
            a = Normal(mean, logstd, sample_dim, n_samples)
        tensors = set(get_backward_tensors(a.value))
        for i in [mean, logstd, sample_dim, n_samples]:
            assert i in tensors
        _ = a.log_prob(np.ones([2, 3]),
                       [mean, logstd, sample_dim, n_samples])


class TestBernoulli:
    def test_init(self):
        with StochasticGraph() as model:
            logits = tf.zeros([2, 3])
            sample_dim = tf.placeholder(tf.int32, shape=())
            n_samples = tf.placeholder(tf.int32, shape=())
            a = Bernoulli(logits, sample_dim, n_samples)
        tensors = set(get_backward_tensors(a.value))
        for i in [logits, sample_dim, n_samples]:
            assert i in tensors
        _ = a.log_prob(np.ones([2, 3]),
                       [logits, sample_dim, n_samples])


class TestDiscrete:
    def test_init(self):
        with StochasticGraph() as model:
            logits = tf.zeros([2, 3])
            sample_dim = tf.placeholder(tf.int32, shape=())
            n_samples = tf.placeholder(tf.int32, shape=())
            a = Discrete(logits, sample_dim, n_samples)
        tensors = set(get_backward_tensors(a.value))
        for i in [logits, sample_dim, n_samples]:
            assert i in tensors
        _ = a.log_prob(np.array([[0, 1, 0], [1, 0, 0]]),
                       [logits, sample_dim, n_samples])
