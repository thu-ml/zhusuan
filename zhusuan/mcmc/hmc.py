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
            return tf.add_n(map(lambda x:
                                0.5*tf.reduce_sum(tf.square(x)),
                                var_list))

        # Half step: p = p + epsilon / 2 * gradient q
        _, grads = get_gradient(current_q)
        current_p = map(lambda (x, y): x + self.step_size/2 * y,
                        zip(current_p, grads))

        # Full steps
        def loop_cond(i, current_q, current_p):
            return i < self.num_leapfrog_steps - 1

        def loop_body(i, current_q, current_p):
            current_q = map(lambda (x, y): x + self.step_size * y,
                            zip(current_q, current_p))

            # p = p + epsilon / 2 * gradient q
            _, grads = get_gradient(current_q)

            current_p = map(lambda (x, y): x + self.step_size * y,
                            zip(current_p, grads))

            return [i+1, current_q, current_p]

        i = tf.constant(0)
        _, current_q, current_p = tf.while_loop(loop_cond,
                             loop_body,
                             [i, current_q, current_p],
                             back_prop=False)

        current_q = map(lambda (x, y): x + self.step_size * y,
                        zip(current_q, current_p))
        _, grads = get_gradient(current_q)
        current_p = map(lambda (x, y): x + self.step_size / 2* y,
                        zip(current_p, grads))

        # Hamiltonian
        old_hamiltonian = -log_posterior(self.q) + kinetic_energy(p)
        new_hamiltonian = -log_posterior(current_q) + kinetic_energy(current_p)

        acceptance_rate = tf.exp(tf.minimum(-new_hamiltonian + old_hamiltonian, 0.0))
        u01 = tf.random_uniform(shape=[])

        new_q = tf.cond(u01 < acceptance_rate,
                        lambda: map(lambda (x, y): x.assign(y),
                                    zip(self.q, current_q)),
                        lambda: self.q
                        )

        if len(self.q) == 1:
            new_q = [new_q]

        return new_q, current_p, old_hamiltonian, new_hamiltonian
