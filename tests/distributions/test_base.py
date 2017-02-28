#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tests.context import zhusuan
from zhusuan.distributions.base import *


class _Dist(Distribution):
    def __init__(self):
        super(_Dist, self).__init__(tf.float32, is_continuous=True)

    def _event_shape(self):
        return tf.constant([3], dtype=tf.int32)

    def _get_event_shape(self):
        return tf.TensorShape([3])

    def _batch_shape(self):
        return tf.constant([2], dtype=tf.int32)

    def _get_batch_shape(self):
        return tf.TensorShape([2])

    def _sample(self, n_samples):
        return tf.ones([n_samples, 2, 3])

    def _log_prob(self, given):
        return tf.reduce_sum(tf.zeros_like(given), -1)

    def _prob(self, given):
        return tf.reduce_prod(tf.ones_like(given), -1)


class TestDistributions(tf.test.TestCase):
    def test_distributions(self):
        dist = _Dist()
