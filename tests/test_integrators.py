#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import pytest

from .context import zhusuan
from zhusuan.mcmc.integrators import *
from zhusuan.mcmc.hamiltonian import *


def test_leapfrog_integrator():
    x = tf.Variable(1.0)
    x_input = tf.placeholder(tf.float32, [])
    update_x = tf.assign(x, x_input)
    log_likelihood = -0.5 * x * x
    gradient = tf.gradients(log_likelihood, x)
    ll_and_grad = [log_likelihood, gradient]

    hamiltonian = Hamiltonian([1])
    epsilon = 0.1

    sess = tf.Session()

    q = [1.0]
    p = [0.0]

    integrator = LeapfrogIntegrator(q, p, epsilon, hamiltonian, [x_input],
                                    [update_x], ll_and_grad, sess, {})

    q0 = q[0]
    p0 = p[0]
    for i in range(10):
        q2, p2, ll, h = integrator.run()

        p0 += 0.5 * epsilon * -q0
        q0 += epsilon * p0
        p0 += 0.5 * epsilon * -q0
        ll0 = -0.5 * q0 * q0
        h0 = -ll0 + 0.5 * p0 * p0

        assert (np.allclose(q2[0], q0))
        assert (np.allclose(p2[0], p0))
        assert (np.allclose(h, h0))
        assert (np.allclose(ll, ll0))
