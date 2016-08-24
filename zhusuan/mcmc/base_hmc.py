#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import time
from zhusuan.diagnostics import ESS
from .hamiltonian import Hamiltonian
from .adapters import StepsizeAdapter, VarianceAdapter
from .integrators import StepsizeTuner


class BaseHMC:
    def __init__(self, sess, data, vars, log_likelihood, epsilon, mass,
                 m_adapt, gamma, t0, kappa, delta,
                 mass_adaptation, init_buffer, base_window, term_buffer):
        np.random.seed(1)

        # Parameters
        self.sess = sess
        self.data = data
        self.log_likelihood = log_likelihood
        self.epsilon = epsilon
        self.delta = delta

        # Variables
        self.q = vars
        self.num_vars = len(self.q)
        shape = map(lambda var: var.initialized_value().get_shape(), self.q)
        self.shape = shape
        self.n_dims = len(shape)

        self.p = map(lambda shape: np.zeros(shape), shape)
        self.q_input = map(lambda shape: tf.placeholder(tf.float32, shape),
                           shape)
        self.update_q = map(lambda (a, b): a.assign(b),
                            zip(self.q, self.q_input))
        self.get_gradients = tf.gradients(self.log_likelihood, self.q)

        # Variables for stepsize adaptation
        self.step = 0
        self.stepsize = StepsizeAdapter(m_adapt, gamma, t0, kappa, delta)
        if mass is not None:
            self.hamiltonian = Hamiltonian(mass)
        else:
            self.hamiltonian = Hamiltonian(
                map(lambda shape: np.ones(shape), shape))

        # Stats
        self.elapsed_time = 0
        self.num_leapfrog = 0
        self.models = []
        self.nL = 0

        # Debug stats
        self.tf_time = 0
        self.tf_grad_time = 0

        # Mass adaptation
        self.mass_adaptation = mass_adaptation
        if mass_adaptation:
            self.mass_adaptor = VarianceAdapter(shape, m_adapt, init_buffer,
                                                base_window, term_buffer,
                                                self.stepsize,
                                                self.hamiltonian)

        # Initialize
        # init = tf.initialize_all_variables()
        # self.sess.run(init)

        # Summary
        # tf.scalar_summary('Loss', self.log_likelihood)
        # tf.scalar_summary('Hamiltonian', finishing_hamiltonian)
        # tf.histogram_summary('W', self.q[0])
        # tf.histogram_summary('pW', self.p[0])
        # self.merged = tf.merge_all_summaries()
        # self.train_writer = tf.train.SummaryWriter('train', sess.graph)

    def stat(self, burnin):
        self.models = np.array(self.models)
        e = ESS(self.models, burnin)
        effectiveness = e / (self.step - burnin)
        grad_per_sample = self.num_leapfrog / e
        grad_per_second = self.num_leapfrog / self.elapsed_time
        second_per_sample = self.elapsed_time / e
        print("Effectiveness = {}%, "
              "{} leapfrog steps per sample ({} grads / s), "
              "{} seconds per sample"
              .format(effectiveness*100, grad_per_sample, grad_per_second,
                      second_per_sample))

    def sample(self):
        time_start = time.time()

        # Initialize the momentum randomly and resample u
        self.p = self.hamiltonian.sample()

        self.q_start = self.sess.run(self.q)
        log_likelihood = self.sess.run(self.log_likelihood,
                                       feed_dict=self.data)
        self.old_hamiltonian = self.hamiltonian.energy(self.p, -log_likelihood)

        if self.stepsize.epsilon_needed:
            self.epsilon = self.stepsize.find_starting_epsilon(
                StepsizeTuner(self.q_start, self.p,
                              self.epsilon, self.hamiltonian, self.delta,
                              self.q_input, self.update_q,
                              [self.log_likelihood, self.get_gradients],
                              self.sess, self.data))

        # Perform sample
        vars, acceptance_rate, trajectory_length = self.sample_work()

        # Finishing
        # Time
        time_end = time.time() - time_start
        self.elapsed_time += time_end

        # Stepsize
        self.step += 1
        self.epsilon = self.stepsize.adapt(acceptance_rate)

        # Mass
        if self.mass_adaptation:
            self.mass_adaptor.adapt(vars)

        # Model (for ESS)
        model = []
        for var in vars:
            model += list(var.flatten())
        self.models.append(model)

        # Print
        print('Step = %d, Trajectory length = %d, acceptance rate = %f, '
              'epsilon = %f' %
              (self.step, trajectory_length, acceptance_rate, self.epsilon))

        return vars

    def sample_work(self):
        pass
