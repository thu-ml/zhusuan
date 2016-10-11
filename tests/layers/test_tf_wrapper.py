#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import six
from mock import Mock
import tensorflow as tf
import numpy as np

from .context import zhusuan
from zhusuan.layers import Layer
from zhusuan.layers.tf_wrapper import _op_book
from zhusuan.layers.tf_wrapper import *


def test_wrappers():
    for k, v in six.iteritems(_op_book):
        assert (v(Mock()).func == getattr(tf, k, None) or
                v(Mock()).func == getattr(tf.nn, k, None))


def test_reduce_sum_wrapper():
    layer = reduce_sum(Layer(Mock()), 1, keep_dims=True)
    output = layer.get_output_for([tf.ones((2, 3))])
    assert(output.get_shape() == (2, 1))
    with tf.Session() as sess:
        output_ = sess.run(output)
        assert(output_.shape == (2, 1))
        assert(np.all((output_) == np.ones((2, 1)) * 3))
