#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os

import tensorflow as tf
import prettytensor as pt
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.utils import log_mean_exp
    from zhusuan.variational import ReparameterizedNormal, advi
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()


class M2(object):
    """
    The deep generative model used in semi-supervised learning with variational
    autoencoders (Kingma, 2014).

    :param n_y: Int. The dimension of latent variables (y), i.e., the number of
        classes.
    :param n_z: Int. The dimension of latent variables (z).
    :param n_x: Int. The dimension of observed variables (x).
    """
    def __init__(self, n_y, n_z, n_x):
        self.n_y = n_y
        self.n_z = n_z
        self.n_x = n_x

    def p_net(self):
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               scale_after_normalization=True):
            l_x_zy = (pt.template('z').join(pt.template('y')).
                      fully_connected(500).
                      batch_normalize().
                      fully_connected(500).
                      batch_normalize().
                      fully_connected(self.n_x, activation_fn=tf.nn.sigmoid))
        return l_x_zy

    def _log_prob(self, z, y, x):
        l_x_zy = (self.p_net().construct(z=tf.reshape(z, (-1, self.n_z)),
                                         y=tf.reshape(y, (-1, self.n_y))).
                  reshape((-1, int(z.get_shape()[1])))).tensor
        log_px_zy = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), l_x_zy, eps=1e-6), 2)
        log_pz = tf.reduce_sum(norm.logpdf(z), 2)
        log_py = tf.log(tf.constant(1., tf.float32) / self.n_y)
        return log_px_zy + log_pz + log_py

    def log_prob(self, latent, observed):
        """
        The log joint probability function.

        :param latent: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_samples, n_latent).
        :param observed: A dictionary of pairs: (string, Tensor). Each of the
            Tensor has shape (batch_size, n_observed).

        :return: A Tensor of shape (batch_size, n_samples). The joint log
            likelihoods.
        """
        raise NotImplementedError()


class M2Unlabeled(M2):
    """
    The M2 model with labels `y` as latent variables.
    """
    def __init__(self, n_y, n_z, n_x):
        super(M2Unlabeled, self).__init__(n_y, n_z, n_x)
        self.l_x_zy = self.p_net()

    def log_prob(self, latent, observed):
        y, z = latent['y'], latent['z']
        x = observed['x']
        return self._log_prob(z, y, x)


class M2Labeled(M2):
    """
    The M2 model with labels `y` as observed variables.
    """
    def __init__(self, n_y, n_z, n_x):
        super(M2Labeled, self).__init__(n_y, n_z, n_x)
        self.l_x_zy = self.p_net()

    def log_prob(self, latent, observed):
        z = latent['z']
        y, x = observed['y'], observed['x']
        return self._log_prob(z, y, x)
