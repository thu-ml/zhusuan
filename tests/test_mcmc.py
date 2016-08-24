#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import pytest
import scipy
import scipy.stats

from .context import zhusuan
from zhusuan.mcmc.nuts import NUTS


def test_nuts():
    x = tf.Variable(tf.zeros([1]))
    log_likelihood = -0.5 * tf.square(x)

    # Data definition
    data = {}

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    n_samples = 1000
    burnin = 500
    sampler = NUTS(sess, data, [x], log_likelihood, m_adapt=burnin,
                   mass_adaptation=True)

    for i in range(n_samples):
        sampler.sample()

    samples = np.squeeze(sampler.models)
    p_value = scipy.stats.normaltest(samples)
    assert(p_value > 0.1)
