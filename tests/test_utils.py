#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from scipy import misc
import numpy as np

from .context import zhusuan
from zhusuan.utils import *


def test_log_sum_exp():
    with tf.Session() as sess:
        a = np.array([[[1., 3., 0.2], [0.7, 2., 1e-6]],
                      [[0., 1e6, 1.], [1., 1., 1.]]])
        for keepdims in [True, False]:
            true_values = misc.logsumexp(a, (0, 2), keepdims=keepdims)
            test_values = sess.run(log_sum_exp(
                tf.constant(a), (0, 2), keepdims))
            assert (np.abs((test_values - true_values) / true_values).max() <
                    1e-6)


def test_log_mean_exp():
    with tf.Session() as sess:
        a = np.array([[[1., 3., 0.2], [0.7, 2., 1e-6]],
                      [[0., 1e6, 1.], [1., 1., 1.]]])
        for keepdims in [True, False]:
            true_values = misc.logsumexp(a, (0, 2), keepdims=keepdims) - \
                          np.log(a.shape[0] * a.shape[2])
            test_values = sess.run(log_mean_exp(
                tf.constant(a), (0, 2), keepdims))
            assert (np.abs((test_values - true_values) / true_values).max() <
                    1e-6)

        b = np.array([[0., 1e-6, 10.1]])
        test_values = sess.run(log_mean_exp(b, 0, keep_dims=False))
        assert (np.abs(test_values - b).max() < 1e-6)
