#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tests.context import zhusuan
from zhusuan.model.stochastic import *
from zhusuan.model.base import StochasticGraph
from zhusuan.model.utils import get_backward_ops


def test_Normal():
    with StochasticGraph() as model:
        mean = tf.zeros([2, 3])
        logstd = tf.zeros([2, 3])
        sample_dim = tf.placeholder(tf.int32, shape=())
        n_samples = tf.placeholder(tf.int32, shape=())
        a = Normal('a', mean, logstd, sample_dim, n_samples)
    ops = set(get_backward_ops(a.tensor))
    for i in [mean, logstd, sample_dim, n_samples]:
        assert i.op in ops
    _ = a.log_prob(np.ones([2, 3]))


def test_Bernoulli():
    with StochasticGraph() as model:
        logits = tf.zeros([2, 3])
        sample_dim = tf.placeholder(tf.int32, shape=())
        n_samples = tf.placeholder(tf.int32, shape=())
        a = Bernoulli('a', logits, sample_dim, n_samples)
    ops = set(get_backward_ops(a.tensor))
    for i in [logits, sample_dim, n_samples]:
        assert i.op in ops
    _ = a.log_prob(np.ones([2, 3]))


def test_Discrete():
    with StochasticGraph() as model:
        logits = tf.zeros([2, 3])
        sample_dim = tf.placeholder(tf.int32, shape=())
        n_samples = tf.placeholder(tf.int32, shape=())
        a = Discrete('a', logits, sample_dim, n_samples)
    ops = set(get_backward_ops(a.tensor))
    for i in [logits, sample_dim, n_samples]:
        assert i.op in ops
    _ = a.log_prob(np.array([[0, 1, 0], [1, 0, 0]]))
