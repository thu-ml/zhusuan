#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from copy import copy


class HMC:
    def __init__(self, step_size=1, num_leapfrog_steps=10):
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps

    def sample(self, log_posterior, var_list=None):
        self.q = copy(var_list)
        self.shapes = map(lambda x: x.initialized_value().get_shape(), self.q)

        # Leapfrog
        p = map(lambda s: tf.random_normal(shape=s), self.shapes)
        current_p = p
        current_q = copy(self.q)

        def get_gradient(var_list):
            log_p = log_posterior(var_list)
            grads = tf.gradients(log_p, var_list)
            return log_p, grads

        def kinetic_energy(var_list):
            sum = 0
            for var in var_list:
                sum += 0.5 * tf.reduce_sum(var * var)

            return sum

        # Half step: p = p + epsilon / 2 * gradient q
        log_p, grads = get_gradient(current_q)
        current_p = map(lambda (x, y): x + self.step_size/2 * y,
                        zip(current_p, grads))

        # Full steps
        for i in range(self.num_leapfrog_steps):
            # q = q + epsilon * p / mass
            current_q = map(lambda (x, y): x + self.step_size * y,
                            zip(current_q, current_p))

            # p = p + epsilon / 2 * gradient q
            step_size = self.step_size / 2 if i + 1 == self.num_leapfrog_steps else self.step_size
            log_p, grads = get_gradient(current_q)
            current_p = map(lambda (x, y): x + step_size * y,
                            zip(current_p, grads))

        # Hamiltonian
        old_hamiltonian = -log_posterior(self.q) + kinetic_energy(p)
        new_hamiltonian = -log_p + kinetic_energy(current_p)

        acceptance_rate = tf.exp(tf.minimum(-new_hamiltonian + old_hamiltonian, 0.0))
        u01 = tf.random_uniform(shape=[])

        new_q = tf.cond(u01 < acceptance_rate,
                        lambda: map(lambda (x, y): x.assign(y), zip(self.q, current_q)),
                        lambda: self.q)

        return new_q, p, old_hamiltonian, new_hamiltonian
