#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from zhusuan.diagnostics import *


class TestESS(tf.test.TestCase):
    def test_ess(self):
        n = 10000
        stride = 1
        dims = 2

        # Gaussian samples
        idepg = np.random.normal(size=(n, dims))
        self.assertTrue(ESS(idepg, burnin=100) >= 2000)

        # Gaussian samples by MCMC
        mcmc = []
        current = np.array([0, 0])

        rate = 0
        for i in range(n):
            next = current + np.random.normal(size=(dims)) * stride
            acceptance_rate = np.exp(
                min(0, -0.5 * np.sum((next ** 2 - current ** 2))))
            if np.random.random() < acceptance_rate:
                current = next
                rate += 1
            mcmc.append(list(current))

        mcmc = np.array(mcmc)
        self.assertTrue(ESS(mcmc, burnin=100) <= 1000)
