#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re
import tensorflow as tf

import zhusuan as zs
from zhusuan.framework.bn import Rule, LazyBayesianNet


def dense(bn, inp, n_out, name, activation=None):
    # TODO: check shape
    n_in = int(inp.get_shape().as_list()[2])
    n_particles = tf.shape(inp)[0]
    w = bn.normal('w' + name,
                  mean=tf.zeros([n_out, n_in + 1]),
                  std=1.,
                  group_ndims=2,
                  n_samples=n_particles,
                  tag='nn/dense')
    h = tf.concat([inp, tf.ones(tf.shape(inp)[:-1])[..., None]], -1)
    h = tf.einsum("imk,ijk->ijm", w, h) / tf.sqrt(tf.to_float(tf.shape(h)[2]))
    if activation is not None:
        h = activation(h)
    return h


def construct_standard_mean_field(bn, name, tag, shape, n_samples):
    w_mean = tf.get_variable(
        'w_mean_' + name, shape=shape, initializer=tf.constant_initializer(0.))
    w_logstd = tf.get_variable(
        'w_logstd_' + name, shape=shape,
        initializer=tf.constant_initializer(0.))
    dist = zs.distributions.Normal(
        w_mean,
        logstd=w_logstd,
        group_ndims=len(shape),
        is_reparameterized=True,
        check_numerics=False)
    return zs.StochasticTensor(bn, name, dist, n_samples=n_samples)
    

def mean_field_for_dense_weights():
    return LazyBayesianNet([Rule(r'nn/dense', construct_standard_mean_field)])
 
