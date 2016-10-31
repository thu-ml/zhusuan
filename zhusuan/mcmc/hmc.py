#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from copy import copy


def leapfrog_integrator(q, p, step_size1, step_size2, grad):
    q = map(lambda (x, y): x + step_size1 * y,
                    zip(q, p))

    # p = p + epsilon / 2 * gradient q
    grads = grad(q)

    p = map(lambda (x, y): x + step_size2 * y,
                    zip(p, grads))

    return (q, p)


def hamiltonian(q, p, log_posterior):
    return -log_posterior(q) + tf.add_n(map(lambda x:
                                0.5*tf.reduce_sum(tf.square(x)),
                                p))


def get_acceptance_rate(old_q, old_p, new_q, new_p, log_posterior):
    old_hamiltonian = hamiltonian(old_q, old_p, log_posterior)
    new_hamiltonian = hamiltonian(new_q, new_p, log_posterior)
    return old_hamiltonian, new_hamiltonian, \
           tf.exp(tf.minimum(-new_hamiltonian + old_hamiltonian, 0.0))


class HMC:
    def __init__(self, step_size=1, num_leapfrog_steps=10, target_acceptance_rate=0.8):
        self.step_size = tf.Variable(step_size)
        self.num_leapfrog_steps = num_leapfrog_steps
        self.target_acceptance_rate=target_acceptance_rate
        #self.t = tf.Variable(0)

    def init_step_size(self, log_posterior, var_list=None):
        factor = 1.1
        self.q = copy(var_list)
        self.shapes = map(lambda x: x.initialized_value().get_shape(), self.q)

        p = map(lambda s: tf.random_normal(shape=s), self.shapes)

        def get_gradient(var_list):
            log_p = log_posterior(var_list)
            grads = tf.gradients(log_p, var_list)
            return log_p, grads

        def loop_cond(step_size, last_acceptance_rate, cond):
            return cond

        def loop_body(step_size, last_acceptance_rate, cond):
            # Calculate acceptance_rate
            new_q, new_p = leapfrog_integrator(self.q, p, tf.constant(0.0), step_size/2,
                                               lambda var_list: get_gradient(var_list)[1])
            new_q, new_p = leapfrog_integrator(new_q, new_p, step_size, step_size/2,
                                               lambda var_list: get_gradient(var_list)[1])
            _, _, acceptance_rate = get_acceptance_rate(self.q, p, new_q, new_p, log_posterior)

            # Change step size and stopping criteria
            new_step_size = tf.cond(tf.less(acceptance_rate, self.target_acceptance_rate),
                                lambda: step_size * (1.0 / factor),
                                lambda: step_size * factor)

            cond = tf.logical_not(
                tf.logical_xor(tf.less(last_acceptance_rate, self.target_acceptance_rate),
                                  tf.less(acceptance_rate, self.target_acceptance_rate)))

            return [tf.Print(new_step_size, [new_step_size, acceptance_rate]),
                    acceptance_rate, cond]

        new_step_size, new_acceptance_rate, _ = tf.while_loop(loop_cond, loop_body,
                            [self.step_size, tf.constant(1.0), tf.constant(True)])

        return tf.assign(self.step_size, new_step_size)

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

        # Full steps
        def loop_cond(i, current_q, current_p):
            return i < self.num_leapfrog_steps + 1

        def loop_body(i, current_q, current_p):
            step_size1 = tf.cond(i > 0,
                                 lambda: self.step_size,
                                 lambda: tf.constant(0.0, dtype=tf.float32))

            step_size2 = tf.cond(tf.logical_and(tf.less(i, self.num_leapfrog_steps),
                                                tf.less(0, i)),
                                lambda: self.step_size,
                                lambda: self.step_size / 2)

            current_q, current_p = leapfrog_integrator(current_q, current_p,
                                                       step_size1, step_size2,
                                                       lambda q: get_gradient(q)[1])

            return [i + 1, current_q, current_p]

        i = tf.constant(0)
        _, current_q, current_p = tf.while_loop(loop_cond,
                             loop_body,
                             [i, current_q, current_p],
                             back_prop=False, parallel_iterations=1)

        # Hamiltonian
        old_hamiltonian, new_hamiltonian, acceptance_rate = \
            get_acceptance_rate(self.q, p, current_q, current_p, log_posterior)
        u01 = tf.random_uniform(shape=[])

        new_q = tf.cond(u01 < acceptance_rate,
                        lambda: map(lambda (x, y): x.assign(y),
                                    zip(self.q, current_q)),
                        lambda: self.q
                        )

        if len(self.q) == 1:
            new_q = [new_q]

        return new_q, p, old_hamiltonian, new_hamiltonian
