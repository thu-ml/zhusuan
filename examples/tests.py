# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import math
import os
import zhusuan
from zhusuan.mcmc.hmc import VarianceEstimator

def test_variance_estimator():
    x = tf.random_normal(shape=[10])
    variance_estimator = VarianceEstimator(shape=x.get_shape())
    op = variance_estimator.add([x])

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    xs = []
    for i in range(100):
        x_sample, o = sess.run([x, op])
        xs.append(x_sample)
        print(o)

    xs = np.squeeze(np.array(xs))

    var_est = sess.run(variance_estimator.variance())
    var = np.var(xs, axis=0)

    print(var_est)
    print(var * 100 / 99)



