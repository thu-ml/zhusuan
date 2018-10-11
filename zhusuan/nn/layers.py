#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re
import tensorflow as tf

import zhusuan as zs


def dense(bn, inp, n_out, name, activation=None):
    assert inp.shape.ndims == 3 and inp.get_shape().as_list()[2] is not None, \
        "input to nn.dense should have the shape \
         [n_samples, batch_size, n_in], and n_in must be static"

    n_in = int(inp.get_shape().as_list()[2])
    n_particles = tf.shape(inp)[0]
    w = bn.normal('w' + name, mean=tf.zeros([n_out, n_in + 1]), std=1.,
                  group_ndims=2, n_samples=n_particles, tag='nn/dense')
    h = tf.concat([inp, tf.ones(tf.shape(inp)[:-1])[..., None]], -1)
    h = tf.einsum("imk,ijk->ijm", w, h) / tf.sqrt(tf.to_float(tf.shape(h)[2]))
    if activation is not None:
        h = activation(h)
    return h

