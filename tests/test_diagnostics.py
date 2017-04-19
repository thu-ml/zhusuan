#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from zhusuan.diagnostics import *


class TestEffectiveSampleSize(tf.test.TestCase):
    def test_effective_sample_size(self):
        rng = np.random.RandomState(1)
        n = 10000
        stride = 1
        dims = 2

        # Gaussian samples
        idepg = rng.normal(size=(n, dims))
        self.assertTrue(effective_sample_size(idepg, burn_in=100) >= 2000)

        # Gaussian samples by MCMC
        mcmc = []
        current = np.array([0, 0])

        rate = 0
        for i in range(n):
            next = current + rng.normal(size=(dims)) * stride
            acceptance_rate = np.exp(
                np.minimum(0, -0.5 * np.sum((next ** 2 - current ** 2))))
            if np.random.random() < acceptance_rate:
                current = next
                rate += 1
            mcmc.append(list(current))

        mcmc = np.array(mcmc)
        self.assertTrue(effective_sample_size(mcmc, burn_in=100) <= 1000)
