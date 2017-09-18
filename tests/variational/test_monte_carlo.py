#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from scipy import stats
import numpy as np
import tensorflow as tf

from zhusuan.variational.monte_carlo import *
from zhusuan.distributions import Normal


class TestImportanceWeightedObjective(tf.test.TestCase):
    def test_objective(self):
        # TODO: test k=1 equal to elbo
        # TODO: test with k increase, increase
        pass

    def test_sgvb(self):
        # TODO: test when k=1, gradients equal to elbo grads
        pass

    def test_vimco(self):
        # TODO: test grads with variance reduction equal to grads without it.
        pass
