#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from scipy import misc

from tests.context import zhusuan
from zhusuan.distributions.utils import *


class TestLogFactorial(tf.test.TestCase):
    def test_log_factorial(self):
        with self.test_session():
            for i in [1, 2, 5, 10]:
                self.assertNear(np.log(misc.factorial(i)),
                                log_factorial(i).eval(), 1e-6)


class TestLogCombination(tf.test.TestCase):
    def test_log_combination(self):
        pass


class TestExplicitBroadcast(tf.test.TestCase):
    def test_explicit_broadcast(self):
        pass
