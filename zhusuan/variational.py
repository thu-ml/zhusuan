#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from .distributions import norm


class Variational(object):
    """
    The :class:`Variational` class represents a variational posterior. It
    should be subclassed when implementing new types of variational posterior.
    """
    def __init__(self):
        pass

    def sample(self, n_samples=1, **kwargs):
        """
        Generate samples from the variational posterior.

        :param n_samples: Int. Number of samples.

        :return: A Tensor of shape (batch_size, n_samples, n_z). Samples from
            the variational posterior.
        """
        raise NotImplementedError()

    def logpdf(self, z, **kwargs):
        """
        The log density function of the variational posterior.

        :param z: A Tensor of shape (batch_size, n_samples, n_z). The value at
            which to evaluate the log density function.

        :return: A Tensor of the same shape as `z` with function values.
        """
        raise NotImplementedError()


class ReparameterizedNormal(Variational):
    """
    The class of variational posterior used in Automatic Differentiation
    Variational Inference (ADVI).
    Note that gradients on samples from this variational posterior are allowed
    to propagate through `vz_mean` and `vz_logstd` in this function, using the
    reparametrization trick from (Kingma, 2013), which is contrary to the
    behavior in `zhusuan.distributions`.

    :param vz_mean: A Tensorflow node that has shape (batch_size, n_z), which
        denotes the mean of the variational posterior to be optimized during
        the inference.
        For traditional mean-field variational inference, the batch_size can
        be set to 1.
        For amortized variational inference, vz_mean depends on x and should
        have the same batch_size as x.
    :param vz_logstd: A Tensorflow node that has shape (batch_size, n_z), which
        denotes the log standard deviation of the variational posterior to be
        optimized during the inference. See `vz_mean` for proper usage.
    """
    def __init__(self, vz_mean, vz_logstd):
        self.vz_mean = vz_mean
        self.vz_logstd = vz_logstd
        super(Variational, self).__init__()

    def sample(self, n_samples=1, **kwargs):
        samples = norm.rvs(
            size=(tf.shape(self.vz_mean)[0], n_samples,
                  tf.shape(self.vz_mean)[1])
        ) * tf.exp(tf.expand_dims(self.vz_logstd, 1)) + tf.expand_dims(
            self.vz_mean, 1)
        samples.set_shape((None, n_samples, None))
        return samples

    def logpdf(self, z, **kwargs):
        return tf.reduce_sum(
            norm.logpdf(z, tf.expand_dims(self.vz_mean, 1),
                        tf.expand_dims(tf.exp(self.vz_logstd), 1)), 2)


def advi(model, x, variational, n_samples=1,
         optimizer=tf.train.AdamOptimizer()):
    """
    Implements the automatic differentiation variational inference (ADVI)
    algorithm. For now we assume all latent variables have been transformed in
    the model definition to have support on R^n.

    :param model: An model object that has a method logprob(z, x) to compute
        the log joint likelihood of the model.
    :param x: 2-D Tensor of shape (batch_size, n_x). Observed data.
    :param variational: A :class:`Variational` object.
    :param n_samples: Int. Number of posterior samples used to
        estimate the gradients. Default to be 1.
    :param optimizer: Tensorflow optimizer object. Default to be
        AdamOptimizer.

    :return: A Tensorflow computation graph of the inference procedure.
    :return: A 0-D Tensor. The variational lower bound.
    """
    samples = variational.sample(n_samples)
    lower_bound = model.log_prob(samples, x) - variational.logpdf(samples)
    lower_bound = tf.reduce_mean(lower_bound)
    return optimizer.minimize(-lower_bound), lower_bound
