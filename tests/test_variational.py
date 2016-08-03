#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import pytest

from .context import zhusuan
from zhusuan.variational import *


class TestVariational():
    def test_init(self):
        Variational()

    def test_sample(self):
        variational = Variational()
        with pytest.raises(NotImplementedError):
            variational.sample()

    def test_logpdf(self):
        variational = Variational()
        with pytest.raises(NotImplementedError):
            variational.logpdf(1.)


class TestReparameterizedNormal():
    def test_init(self):
        vz_mean = np.ones(3)
        vz_logstd = np.zeros(3)
        with pytest.raises(ValueError):
            ReparameterizedNormal(vz_mean, vz_logstd)

        with tf.Session() as sess:
            vz_mean = tf.placeholder(tf.float32)
            vz_logstd = np.zeros((1, 3))
            variational = ReparameterizedNormal(vz_mean, vz_logstd)
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(variational.vz_mean,
                         feed_dict={vz_mean: np.ones((3))})

            variational = ReparameterizedNormal(tf.Variable(np.ones((1, 4))),
                                                np.ones((5, 4)))
            sess.run(variational.vz_logstd)

    def test_sample(self):
        pass

    def test_logpdf(self):
        pass


def test_advi():
    pass
