#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
from zhusuan.utils import copy, if_raise


class LeapfrogIntegrator:
    """
    The Leapfrog integrator simulates the movement of `(q, p)` in a dynamics
    `hamiltonian`.

    :param q: Initial position.
    :param p: Initial momentum.
    :param epsilon: Step size (can be negative).
    :param hamiltonian: The Hamiltonian dynamics.
    :param q_input: TensorFlow placeholder for changing `q`.
    :param update_q: TensorFlow op for changing `q`.
    :param ll_and_grad: TensorFlow op that returns log probability and its
    gradient with respect to `q`.
    :param sess: TensorFlow session.
    :param data: Data (feed_dict) to feed into TensorFlow.
    """
    def __init__(self, q, p, epsilon, hamiltonian, q_input, update_q,
                 ll_and_grad, sess, data):
        self.epsilon = epsilon
        self.hamiltonian = hamiltonian
        self.q = copy(q)
        self.p = copy(p)
        self.q_out = copy(q)
        self.p_out = copy(p)

        self.q_input = q_input
        self.sess = sess
        self.update_q = update_q
        self.ll_and_grad = ll_and_grad
        self.data = data

        self.leapfrog(0, epsilon / 2, epsilon)

    def run(self):
        """
        Simulate the dynamics for one step.

        :return: A quadruple of (new q, new p, new log probability,
        new hamiltonian)
        """
        log_likelihood, hamiltonian = self.leapfrog(self.epsilon / 2,
                                                    self.epsilon,
                                                    self.epsilon)
        return self.q_out, self.p_out, log_likelihood, hamiltonian

    def leapfrog(self, epsilon1, epsilon2, epsilon3):
        self.sess.run(self.update_q, feed_dict={a: b for a, b in
                                                zip(self.q_input, self.q)})
        log_likelihood, gradients = self.sess.run(self.ll_and_grad,
                                                  feed_dict=self.data)

        for i in range(len(self.q)):
            self.p_out[i] = self.p[i] + epsilon1 * gradients[i]
            self.q_out[i] = np.copy(self.q[i])

            self.p[i] += epsilon2 * gradients[i]

        velocity = self.hamiltonian.velocity(self.p)
        for i in range(len(self.q)):
            self.q[i] += epsilon3 * velocity[i]

        return log_likelihood, \
            self.hamiltonian.energy(self.p_out, -log_likelihood)


class StepsizeTuner:
    """
    The :class:`StepsizeTuner` class returns a step size where the
    acceptance rate of a Langevin step from `(q, p)` just crosses delta.

    :param q: Initial position.
    :param p: Initial momentum.
    :param epsilon: Step size (can be negative).
    :param hamiltonian: The Hamiltonian dynamics.
    :param delta: The target acceptance rate.
    :param q_input: TensorFlow placeholder for changing `q`.
    :param update_q: TensorFlow op for changing `q`.
    :param ll_and_grad: TensorFlow op that returns log probability and its
    gradient with respect to `q`.
    :param sess: TensorFlow session.
    :param data: Data (feed_dict) to feed into TensorFlow.
    """
    def __init__(self, q, p, epsilon, hamiltonian, delta, q_input, update_q,
                 ll_and_grad, sess, data):
        self.epsilon = epsilon
        self.hamiltonian = hamiltonian
        self.delta = delta
        self.q = q
        self.p = p
        self.n_vars = len(self.q)

        self.q_input = q_input
        self.sess = sess
        self.update_q = update_q
        self.ll_and_grad = ll_and_grad
        self.data = data

        self.sess.run(self.update_q, feed_dict={a: b for a, b in
                                                zip(self.q_input, q)})
        ll, self.gradients = self.sess.run(ll_and_grad, feed_dict=self.data)
        self.old_hamiltonian = self.hamiltonian.energy(p, -ll)
        if_raise(np.isnan(self.old_hamiltonian),
                 RuntimeError('Hamiltonian is nan'))

    def tune(self):
        """
        Tune the step size for the target acceptance rate.

        :return: The step size.
        """
        current_epsilon = self.epsilon
        is_large = None
        factor = 1.1
        while True:
            # Initialize
            q = copy(self.q)
            p = copy(self.p)

            # Leapfrog p
            for i in range(self.n_vars):
                p[i] += current_epsilon / 2 * self.gradients[i]

            # Leapfrog q
            velocity = self.hamiltonian.velocity(p)
            for i in range(self.n_vars):
                q[i] += current_epsilon * velocity[i]

            # Leapfrog p
            self.sess.run(self.update_q, feed_dict={a: b for a, b in
                                                    zip(self.q_input, q)})

            log_likelihood, gradients = self.sess.run(self.ll_and_grad,
                                                      feed_dict=self.data)
            for i in range(self.n_vars):
                p[i] += current_epsilon / 2 * gradients[i]

            new_hamiltonian = self.hamiltonian.energy(p, -log_likelihood)

            # Update
            acceptance_rate = np.exp(
                min(0, -(new_hamiltonian - self.old_hamiltonian)))
            print('Epsilon = {}, acceptance_rate = {}'.format(current_epsilon,
                                                              acceptance_rate))

            acceptance_rate = 0 if np.isnan(acceptance_rate) \
                else acceptance_rate

            current_is_large = acceptance_rate > self.delta
            if is_large is None:
                is_large = current_is_large
            elif is_large:
                if not current_is_large:
                    return current_epsilon / factor
                else:
                    current_epsilon *= factor
            else:
                if current_is_large:
                    return current_epsilon
                else:
                    current_epsilon /= factor
