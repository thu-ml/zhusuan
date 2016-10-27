#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from zhusuan.utils import add_name_scope
from zhusuan.distributions import norm, discrete, bernoulli
from .base import *


class Normal(StochasticTensor):
    def __init__(self,
                 mean,
                 logstd,
                 sample_dim=None,
                 n_samples=1,
                 reparameterized=True,
                 name=None):
        super(Normal, self).__init__([mean, logstd, sample_dim, n_samples])

    @add_name_scope
    def sample(self, **kwargs):
        pass

    @add_name_scope
    def log_p(self, given, inputs):
        pass


class Discrete(StochasticTensor):
    def __init__(self, p, sample_index=None, n_samples=1, name=None):
        super(Discrete, self).__init__([p, sample_index, n_samples])

    @add_name_scope
    def sample(self, **kwargs):
        pass

    @add_name_scope
    def log_p(self, given, inputs):
        pass


class Bernoulli(StochasticTensor):
    def __init__(self, p, sample_index=None, n_samples=1, name=None):
        super(Bernoulli, self).__init__([p, sample_index, n_samples])

    @add_name_scope
    def sample(self, **kwargs):
        pass

    @add_name_scope
    def log_p(self, given, inputs):
        pass


# class Normal(Layer):
#     """
#     The :class:`Normal` class represents a Normal distribution
#     layer that accepts the mean and the log standard deviation as inputs, which
#     is used in Automatic Differentiation Variational Inference (ADVI).
#
#     :param incomings: A list of 2 :class:`Layer` instances. The first
#         representing the mean, and the second representing the log variance.
#         Must be of shape (>=3)-D like (batch_size, n_samples, ...).
#     :param reparameterized: Bool. If True, gradients on samples from this
#         Normal distribution are allowed to propagate into inputs in this
#         function, using the reparametrization trick from (Kingma, 2013).
#     :param n_samples: Int or a scalar Tensor of type int. Number of samples
#         drawn for distribution layers. Default to be 1.
#     :param name: A string or None. An optional name to attach to this layer.
#     """
#     def __init__(self, incomings, n_samples=1, reparameterized=True,
#                  name=None):
#         super(Normal, self).__init__(incomings, name)
#         if len(incomings) != 2:
#             raise ValueError("Normal layer only accepts input "
#                              "layers of length 2 (the mean and the log "
#                              "standard deviation).")
#         if isinstance(n_samples, int):
#             self.n_samples = n_samples
#         else:
#             with tf.control_dependencies([tf.assert_rank(n_samples, 0)]):
#                 self.n_samples = tf.cast(n_samples, tf.int32)
#         self.reparameterized = reparameterized
#
#     @add_name_scope
#     def get_output_for(self, inputs, deterministic=False, **kwargs):
#         mean_, logvar = inputs
#         if deterministic:
#             return mean_
#
#         samples_1 = norm.rvs(shape=tf.shape(mean_)) * tf.exp(0.5 * logvar) + \
#             mean_
#         samples_n = norm.rvs(
#             shape=tf.concat(0, [tf.pack([tf.shape(mean_)[0], self.n_samples]),
#                                 tf.shape(mean_)[2:]])
#             ) * tf.exp(0.5 * logvar) + mean_
#
#         def _output():
#             if isinstance(self.n_samples, int):
#                 if self.n_samples == 1:
#                     return samples_1
#                 else:
#                     samples_n.set_shape([None, self.n_samples] +
#                                         [None] * len(mean_.get_shape()[2:]))
#                     return samples_n
#             else:
#                 return tf.cond(tf.equal(self.n_samples, 1), lambda: samples_1,
#                                lambda: samples_n)
#
#         if self.reparameterized:
#             return _output()
#         else:
#             return tf.stop_gradient(_output())
#
#     @add_name_scope
#     def get_logpdf_for(self, output, inputs, **kwargs):
#         mean_, logvar = inputs
#         return tf.reduce_sum(
#             norm.logpdf(output, mean_, tf.exp(0.5 * logvar)),
#             list(range(2, len(mean_.get_shape()))))
#
#
# class Discrete(Layer):
#     """
#     The :class:`Discrete` class represents a discrete distribution layer that
#     accepts the class probabilities as inputs.
#
#     :param incoming: A :class:`Layer` instance. The layer feeding into this
#         layer which gives output as class probabilities. Must be of shape 3-D
#         like (batch_size, n_samples, n_dims).
#     :param n_samples: Int or a Tensor of type int. Number of samples drawn for
#         distribution layers. Default to be 1.
#     :param n_classes: Int. Number of classes for this discrete distribution.
#         Should be the same as the 3rd dimension of the incoming.
#     :param name: A string or None. An optional name to attach to this layer.
#     """
#     def __init__(self, incoming, n_classes, n_samples=1, name=None):
#         super(Discrete, self).__init__(incoming, name)
#         if isinstance(n_samples, int):
#             self.n_samples = n_samples
#         else:
#             with tf.control_dependencies([tf.assert_rank(n_samples, 0)]):
#                 self.n_samples = tf.cast(n_samples, tf.int32)
#         self.n_classes = n_classes
#
#     def _static_shape_check(self, input):
#         static_n_dim = input.get_shape().as_list()[2]
#         if (static_n_dim is not None) and (static_n_dim != self.n_classes):
#             raise (ValueError("Input fed into Discrete layer %r has different "
#                               "number of classes with the argument n_classes "
#                               "passed on construction of this layer." % self))
#
#     @add_name_scope
#     def get_output_for(self, input, deterministic=False, **kwargs):
#         self._static_shape_check(input)
#         if deterministic:
#             return input
#
#         n_dim = tf.shape(input)[2]
#         samples_1_2d = discrete.rvs(tf.reshape(input, (-1, n_dim)))
#         samples_1 = tf.reshape(samples_1_2d, tf.shape(input))
#         samples_1.set_shape(input.get_shape())
#
#         samples_n_2d = discrete.rvs(
#             tf.reshape(tf.tile(input, (1, self.n_samples, 1)), (-1, n_dim)))
#         samples_n = tf.reshape(samples_n_2d, (-1, self.n_samples,
#                                               tf.shape(samples_n_2d)[1]))
#         samples_n.set_shape([None, None, self.n_classes])
#
#         if isinstance(self.n_samples, int):
#             if self.n_samples == 1:
#                 return samples_1
#             else:
#                 return samples_n
#         else:
#             return tf.cond(tf.equal(self.n_samples, 1), lambda: samples_1,
#                            lambda: samples_n)
#
#     @add_name_scope
#     def get_logpdf_for(self, output, input, **kwargs):
#         self._static_shape_check(input)
#         output_, input_ = ensure_dim_match([output, input], 1)
#         output_2d = tf.reshape(output_, (-1, tf.shape(output_)[2]))
#         input_2d = tf.reshape(input_, (-1, tf.shape(input_)[2]))
#         ret = tf.reshape(discrete.logpdf(output_2d, input_2d),
#                          (-1, tf.shape(output_)[1]))
#         ret.set_shape(output_.get_shape().as_list()[:2])
#         return ret
