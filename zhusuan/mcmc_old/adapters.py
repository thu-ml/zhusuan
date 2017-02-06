#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
from zhusuan.utils import VarianceEstimator


class StepsizeAdapter:
    """
    The :class:`StepsizeAdapter` class implements Nestreov's dual averaging
    method for adjusting step size.

    :param m_adapt: Number of adapting steps.
    :param gamma: Weight of stochastic approximation term.
    :param t0: Initial time.
    :param kappa: Decay rate.
    :param delta: Target acceptance rate
    """
    def __init__(self, m_adapt=50, gamma=0.05, t0=10, kappa=0.75, delta=0.8):
        self.log_epsilon = 0
        self.mu = 0

        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.m_adapt = m_adapt
        self.delta = delta

        self.step = 0
        self.total_step = 0
        self.log_epsilon_bar = 0
        self.h_bar = 0
        self.epsilon_needed = True

    def restart(self):
        """
        Restart the adapter.
        """
        self.step = 0
        self.h_bar = 0
        self.log_epsilon_bar = 0
        self.epsilon_needed = True

    def find_starting_epsilon(self, tuner):
        """
        Find a new starting epsilon using `tuner`.

        :param tuner: The stepsize tuner.
        """
        self.epsilon_needed = False
        epsilon = tuner.tune()
        self.log_epsilon = np.log(epsilon)
        self.mu = np.log(10 * epsilon)
        return epsilon

    def adapt(self, acceptance_rate):
        """
        Adapt the stepsize with respect to the given acceptance rate

        :param acceptance_rate: The given acceptance rate.
        :return: New stepsize.
        """
        self.step += 1
        self.total_step += 1
        # Dual averaging
        if self.total_step < self.m_adapt:
            rate1 = 1. / (self.step + self.t0)
            self.h_bar = (1 - rate1) * self.h_bar \
                + rate1 * (self.delta - acceptance_rate)
            # print('mu = {}, H_bar = {}'.format(self.mu, self.h_bar))
            self.log_epsilon = self.mu - \
                np.sqrt(self.step)/self.gamma * self.h_bar
            rate = self.step ** (-self.kappa)
            self.log_epsilon_bar = rate * self.log_epsilon + \
                (1-rate) * self.log_epsilon_bar

            return np.exp(self.log_epsilon)
        else:
            return np.exp(self.log_epsilon_bar)


class VarianceAdapter:
    """
    The :class:`VarianceAdapter` class automatically tunes the mass matrix
    for HMC, using the variance of the position `q`.

    :param shape: Shape of position (list of shapes).
    :param burnin: Number of burnin iterations.
    :param init_buffer: Size of initialize buffer, which do not accumulate
    samples for variance.
    :param base_window: Period to update the mass matrix.
    :param term_buffer: Size of terminating buffer, which only adapts the
    stepsize.
    :param stepsize: A stepsize adapter class (we will inform it to restart).
    :param hamiltonian: A Hamiltonian dynamics (we will update its mass
    matrix).
    """
    def __init__(self, shape, burnin, init_buffer, base_window, term_buffer,
                 stepsize, hamiltonian):
        self.shape = shape
        self.step = 0

        self.burnin = burnin

        self.init_buffer = init_buffer
        self.base_window = base_window
        self.term_buffer = term_buffer

        self.estimator = VarianceEstimator(shape)

        self.stepsize = stepsize
        self.hamiltonian = hamiltonian

        if init_buffer + term_buffer + base_window > burnin:
            print("Warning: the init buffer, term buffer and base window "
                  "overflow the total number of burnin iterations. Using "
                  "a 20%/30%/50% partition.")

            self.init_buffer = int(burnin * 0.2)
            self.term_buffer = int(burnin * 0.5)
            self.base_window = burnin - self.init_buffer - self.term_buffer

        self.next_window = self.init_buffer + self.base_window

    def adapt(self, q):
        """
        Adapt the mass with the current position.

        :param q: The current position.
        """
        self.step += 1
        if self.step >= self.init_buffer and \
                self.step < self.burnin - self.term_buffer:
            self.estimator.add(q)

        if self.step >= self.burnin:
            return

        if self.step == self.next_window:
            print('Estimating variance...')
            self.base_window *= 2
            self.next_window += self.base_window

            num_samples = self.estimator.count
            rate = float(num_samples) / (num_samples + 5)
            precision = map(lambda x: 1 / (rate*x + 1e-3*(1-rate)),
                            self.estimator.variance())

            self.stepsize.restart()
            self.hamiltonian.mass = precision
