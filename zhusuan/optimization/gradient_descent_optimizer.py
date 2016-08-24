#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from zhusuan.utils import copy, if_raise


class GradientDescentOptimizer:
    """
    A simple batch gradient descent with a line search satisfying the
    Armijo-Goldstein condition.
    """
    def __init__(self, sess, data, objective, vars, init=None,
                 stepsize=1, max_n_iterations=100, tol=1e-3, stepsize_tol=1e-7,
                 c=0.5, tau=1.2):
        """

        :param sess:
        :param data:
        :param objective:
        :param vars:
        :param init:
        :param stepsize:
        :param max_n_iterations:
        :param tol:
        :param stepsize_tol:
        :param c:
        :param tau:
        """
        self.sess = sess
        self.data = data
        self.objective = objective
        self.vars = vars
        self.shape = map(lambda var: var.initialized_value().get_shape(), vars)
        self.vars_input = map(lambda shape:
                              tf.placeholder(tf.float32, shape), self.shape)
        self.update_vars = map(lambda (x, y): x.assign(y),
                               zip(vars, self.vars_input))

        if init is None:
            self.x = map(lambda shape: np.zeros(shape), self.shape)
        else:
            self.x = copy(init)

        self.stepsize = stepsize
        self.max_n_iterations = max_n_iterations
        self.tol = tol
        self.stepsize_tol = stepsize_tol

        # Parameters for line searching
        self.c = c
        self.tau = tau

        self.get_gradients = tf.gradients(objective, vars)

    def get_obj_and_grad(self):
        self.sess.run(self.update_vars, feed_dict={
            a: b for a, b in zip(self.vars_input, self.x)})
        obj, g = self.sess.run([self.objective, self.get_gradients],
                               feed_dict=self.data)
        return obj, g

    def get_obj(self, new_x):
        self.sess.run(self.update_vars, feed_dict={a: b for a, b in
                                                   zip(self.vars_input,
                                                       new_x)})
        obj = self.sess.run(self.objective,
                            feed_dict=self.data)
        return obj

    def stop(self, new_obj, obj):
        rel_change = abs(new_obj / obj - 1)
        if_stop = rel_change < self.tol or abs(new_obj - obj) < self.tol
        return if_stop

    def optimize(self):
        for i in range(self.max_n_iterations):
            self.stepsize *= self.tau

            obj, g = self.get_obj_and_grad()
            # print('Objective = {}, gradient = {}'.format(obj, g))
            # print('X = {}'.format(self.x))
            if_raise(np.isnan(obj),
                     RuntimeError('Objective is nan, consider specifing '
                                  'initializing value.'))

            t = self.c * np.linalg.norm(g) ** 2

            # Perform line search
            while True:
                new_x = map(lambda (x, y): x-self.stepsize*y, zip(self.x, g))
                new_obj = self.get_obj(new_x)
                new_obj = 1e100 if np.isnan(new_obj) else new_obj
                # print('Stepsize = {}, New point = {}, New objective = {}'
                #      .format(self.stepsize, new_x, new_obj))

                amount_of_decrease = obj - new_obj
                if amount_of_decrease > self.stepsize * t \
                   or self.stepsize < self.stepsize_tol:
                    self.x = new_x
                    break
                self.stepsize /= self.tau

            if self.stop(new_obj, obj):
                break

            print('Step {}, objective = {}, stepsize = {}'
                  .format(i, new_obj, self.stepsize))

        return self.x
