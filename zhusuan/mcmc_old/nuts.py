#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import math
import numpy as np
from zhusuan.utils import copy, MeanStatistics, if_raise
from .integrators import LeapfrogIntegrator
from .candidate import Candidate
from .base_hmc import BaseHMC

max_depth = 100


class NUTS(BaseHMC):
    """
    The No-U-Turn sampler.

    :param sess: A TensorFlow session.
    :param data: The data (feed_dict) to feed in.
    :param log_likelihood: The log probability of the model.
    :param epsilon: Initial step size (default 0.1).
    :param mass: (Optional) Initial diagonal mass (default None).
    :param m_adapt: Number of adapting steps (default 50).
    :param gamma: Weight of stochastic approximation term (default 0.05).
    :param t0: Initial time (default 10).
    :param kappa: Decay rate (default 0.75).
    :param delta: Target acceptance rate (default 0.8).
    :param mass_adaptation: Whether adapting mass (default False).
    :param init_buffer: Size of initialize buffer, which do not accumulate
    samples for variance (default 15).
    :param base_window: Period to update the mass matrix (default 30).
    :param term_buffer: Size of terminating buffer, which only adapts the
    stepsize (default 50).
    """

    def __init__(self, sess, data, vars, log_likelihood, epsilon=0.1,
                 mass=None,
                 m_adapt=50, gamma=0.05, t0=10, kappa=0.75, delta=0.8,
                 mass_adaptation=False, init_buffer=15,
                 base_window=30, term_buffer=50):
        BaseHMC.__init__(self, sess, data, vars, log_likelihood, epsilon, mass,
                         m_adapt, gamma, t0, kappa, delta,
                         mass_adaptation, init_buffer,
                         base_window, term_buffer)

        # Slice sampling
        self.delta_max = 1000

        # Stack
        self.max_depth = 0
        self.q_stack = []
        self.p_stack = []
        self.enlarge_stack()

        # Candidate
        self.local_candidate = Candidate(shape=self.shape)

        # Acceptance
        self.mean_acceptance_rate = MeanStatistics()

    def sample_work(self):
        """
        Internal function for obtaining a sample.
        """
        log_u = math.log(np.random.random()) - self.old_hamiltonian

        if_raise(np.isnan(self.old_hamiltonian),
                 RuntimeError(
                     "Hamiltonian is nan, maybe because stepsize is too large"
                     ", the network does not handle numerical extreme cases,"
                     " or the starting point is invalid.")
                 )

        # Initialize integrators
        right_integrator = LeapfrogIntegrator(self.q_start, self.p,
                                              self.epsilon,
                                              self.hamiltonian,
                                              self.q_input, self.update_q,
                                              [self.log_likelihood,
                                               self.get_gradients],
                                              self.sess, self.data)
        left_integrator = LeapfrogIntegrator(self.q_start, self.p,
                                             -self.epsilon,
                                             self.hamiltonian,
                                             self.q_input, self.update_q,
                                             [self.log_likelihood,
                                              self.get_gradients],
                                             self.sess, self.data)

        # Initialize candidate and stack
        self.global_candidate = Candidate(init=self.q_start)
        self.mean_acceptance_rate.reset()
        self.mean_acceptance_rate.add(1.0)

        # Start NUTS
        old_num_leapfrog = self.num_leapfrog
        for depth in range(max_depth):
            if self.max_depth < depth + 1:
                self.enlarge_stack()

            self.local_candidate.reset()
            is_right = np.random.random() < 0.5
            if is_right:
                self.q_stack[0] = copy(left_integrator.q_out)
                self.p_stack[0] = copy(left_integrator.p_out)
                legal_tree, root_legal = self.build_tree(depth,
                                                         right_integrator, 1,
                                                         self.old_hamiltonian,
                                                         log_u)
            else:
                self.q_stack[0] = copy(right_integrator.q_out)
                self.p_stack[0] = copy(right_integrator.p_out)
                legal_tree, root_legal = self.build_tree(depth,
                                                         left_integrator, -1,
                                                         self.old_hamiltonian,
                                                         log_u)

            if not legal_tree:
                break

            self.global_candidate.merge(self.local_candidate)

            if not root_legal:
                break

        # Finish
        trajectory_length = self.num_leapfrog - old_num_leapfrog
        # if depth == max_depth:
        #     print('Warning: max depth has been reached')

        self.sess.run(self.update_q, feed_dict={a: b for a, b in zip(
            self.q_input, self.global_candidate.x)})

        return self.global_candidate.x, \
            self.mean_acceptance_rate.mean(), trajectory_length

    def u_turn(self, q, p, depth, direction):
        """
        Check if there are u-turns
        """
        sum_1 = 0
        sum_2 = 0
        for i in range(self.num_vars):
            delta_q = q[i] - self.q_stack[depth][i]
            sum_1 += np.sum(delta_q * p[i])
            sum_2 += np.sum(delta_q * self.p_stack[depth][i])

        return sum_1 * direction < 0 or sum_2 * direction < 0, min(
            sum_1 * direction, sum_2 * direction)

    def build_tree(self, depth, integrator, direction, old_hamiltonian, log_u):
        """
        Builds the tree.

        :param depth: Tree depth.
        :param integrator: Integrator.
        :param direction: Direction of time, +1 or -1.
        :param old_hamiltonian: Old Hamiltonian.
        :param log_u: logarithm of u.
        :return: whether the tree is legal, and whether the root is legal
        """
        legal_tree = True
        root_u_turn = False

        for i in range(2 ** depth):
            # Integrator
            q, p, log_likelihood, new_hamiltonian = integrator.run()
            if_raise(np.isnan(log_likelihood) or np.isnan(new_hamiltonian),
                     RuntimeError(
                         "Hamiltonian is nan, maybe because stepsize is too "
                         "large or the network does not handle numerical "
                         "extreme cases.")
                     )

            # Debug
            self.num_leapfrog += 1
            # print(log_likelihood, new_hamiltonian, is_candidate, direction)

            # Check u-turn
            u_turn = False
            if i == 2 ** depth - 1:
                root_u_turn, f = self.u_turn(q, p, 0, direction)

            current_i = i
            for d in range(depth, 0, -1):
                if current_i % 2 == 0:
                    break
                u_turn, f = self.u_turn(q, p, d, direction)
                if u_turn:
                    break
                current_i //= 2

            # Update stack
            current_i = i
            for d in range(depth, 0, -1):
                if current_i % 2 == 1:
                    break
                self.p_stack[d] = copy(p)
                self.q_stack[d] = copy(q)
                current_i //= 2

            # Check candidate
            is_candidate = log_u < -new_hamiltonian
            is_legal = log_u < -new_hamiltonian + self.delta_max

            if is_candidate:
                self.local_candidate.add(q)

            acceptance_rate = math.exp(
                min(0, -(new_hamiltonian - old_hamiltonian)))
            self.mean_acceptance_rate.add(acceptance_rate)

            if not is_legal or u_turn:
                if not is_legal:
                    print('Exit because of discretization error {}, {}, {}'
                          .format(new_hamiltonian, old_hamiltonian,
                                  acceptance_rate))
                else:
                    print('Exit because of u turn')

                return False, False

        return True, not root_u_turn

    def enlarge_stack(self):
        """
        Enlarge the stack.
        """
        self.q_stack.append(copy(self.p))
        self.p_stack.append(copy(self.p))
