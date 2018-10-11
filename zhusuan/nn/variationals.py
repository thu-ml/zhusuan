#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re
import tensorflow as tf

import zhusuan as zs
from zhusuan.framework.bn import Rule, LazyBayesianNet


def construct_standard_mean_field(bn, name, tag, shape, n_samples):
    w_mean = tf.get_variable(
        name + '_mean', shape=shape, initializer=tf.constant_initializer(0.))
    w_logstd = tf.get_variable(
        name + '_logstd', shape=shape,
        initializer=tf.constant_initializer(0.))
    dist = zs.distributions.Normal(
        w_mean,
        logstd=w_logstd,
        group_ndims=len(shape),
        is_reparameterized=True,
        check_numerics=False)
    # When bn.stochastic look up name in meta_bn, it uses bn._local_cxt, which
    # is assigned in the lazy bn's constructor, as intended
    return bn.stochastic(name, dist, tag=tag, n_samples=n_samples)


def mean_field_for_dense_weights():
    return LazyBayesianNet([Rule(r'nn/dense', construct_standard_mean_field)])
 
