#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from tests.context import zhusuan
from zhusuan.mcmc_old.optimization import GradientDescentOptimizer


def test_gradient_descent_optimizer():
    x = tf.Variable(0.0)
    func = (x + 1) * (x + 1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    optimizer = GradientDescentOptimizer(sess, {}, func, [x], init=[0.0],
                                         stepsize=1, tol=1e-7)
    optimizer = GradientDescentOptimizer(sess, {}, func, [x],
                                         stepsize=1, tol=1e-7)
    x = optimizer.optimize()
    assert abs(x[0] + 1) < 1e-3
