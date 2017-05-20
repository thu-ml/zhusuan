#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from zhusuan.utils import convert_to_int
import numpy as np


__all__ = [
    'planar_normalizing_flow',
]


def linear_ar(name, id, z, hidden=None):
    """
    Implement the linear autoregressive network for Inverse Autoregressive
    Flow (:func:`inv_autoregressive_flow`).

    :param name: A string, to define the parameters' name scope.
    :param id: A int, to define the parameters' name scope.
    :param z: A N-D `float32` Tensor, the original samples.
    :param hidden: Linear autoregressive flow doesn't need hidden layer.

    :return: A N-D Tensor, `m` in the paper (Kingma 2016).
    :return: A N-D Tensor, `s` in the paper (Kingma 2016).
    """
    input_x = tf.convert_to_tensor(z, dtype=tf.float32)
    static_x_shape = input_x.get_shape()
    if not static_x_shape[-1:].is_fully_defined():
        raise ValueError(
            'Inputs {} has unknown static shape in the last axis.'.
            format(input_x))
    d = int(static_x_shape[-1])

    mask = []
    for i in range(d):
        mask_i = [0] * d
        for j in range(d):
            mask_i[j] = int(i < j) * 1.0
        mask.append(mask_i)
    mask = tf.constant(mask, dtype=tf.float32)

    z = tf.reshape(input_x, [-1, d])
    with tf.name_scope(name + '%d' % id):
        m_w = tf.Variable(
            tf.random_normal(shape=[d, d], mean=0, stddev=0.005,
                             dtype=tf.float32),
            name='m_w')
        s_w = tf.Variable(
            tf.random_normal(shape=[d, d], mean=0, stddev=0.005,
                             dtype=tf.float32),
            name='s_w')
        m_w = mask * m_w
        s_w = mask * s_w

        m = tf.matmul(z, m_w)
        s = tf.matmul(z, s_w)

        s = tf.exp(s)

        m = tf.reshape(m, tf.shape(input_x))
        s = tf.reshape(s, tf.shape(input_x))

    return m, s


def planar_normalizing_flow(samples, log_probs, n_iters):
    """
    Perform Planar Normalizing Flow along the last axis of inputs.

    .. math ::

        f(z_t) = z_{t-1} + h(z_{t-1} * w_t + b_t) * u_t

    with activation function `tanh` as well as the invertibility trick
    from (Danilo 2016).

    :param samples: A N-D (N>=2) `float32` Tensor of shape `[..., d]`, and
        planar normalizing flow will be performed along the last axis.
    :param log_probs: A (N-1)-D `float32` Tensor, should be of the same shape
        as the first N-1 axes of `samples`.
    :param n_iters: A int, which represents the number of successive flows.

    :return: A N-D Tensor, the transformed samples.
    :return: A (N-1)-D Tensor, the log probabilities of the transformed
        samples.
    """
    samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)

    if not isinstance(n_iters, int):
        raise ValueError('n_iters should be type \'int\'')

    # check shapes of samples and log_probs
    static_sample_shape = samples.get_shape()
    static_logprob_shape = log_probs.get_shape()
    static_sample_ndim = convert_to_int(static_sample_shape.ndims)
    static_logprob_ndim = convert_to_int(static_logprob_shape.ndims)
    if static_sample_ndim and static_sample_ndim <= 1:
        raise ValueError('samples should have rank >= 2')
    if static_sample_ndim and static_logprob_ndim \
            and static_sample_ndim != static_logprob_ndim + 1:
        raise ValueError('log_probs should have rank (N-1), while N is the '
                         'rank of samples')
    try:
        tf.broadcast_static_shape(static_sample_shape[:-1],
                                  static_logprob_shape)
    except ValueError:
        raise ValueError(
             "samples and log_probs don't have same shape of (N-1) dims,"
             "while N is the rank of samples")
    dynamic_sample_shape = tf.shape(samples)
    dynamic_logprob_shape = tf.shape(log_probs)
    dynamic_sample_ndim = tf.rank(samples)
    dynamic_logprob_ndim = tf.rank(log_probs)
    _assert_sample_ndim = \
        tf.assert_greater_equal(dynamic_sample_ndim, 2,
                                message='samples should have rank >= 2')
    with tf.control_dependencies([_assert_sample_ndim]):
        samples = tf.identity(samples)
    _assert_logprob_ndim = \
        tf.assert_equal(dynamic_logprob_ndim, dynamic_sample_ndim - 1,
                        message='log_probs should have rank (N-1), while N is'
                                ' the rank of samples')
    with tf.control_dependencies([_assert_logprob_ndim]):
        log_probs = tf.identity(log_probs)
    _assert_same_shape = \
        tf.assert_equal(dynamic_sample_shape[:-1], dynamic_logprob_shape,
                        message="samples and log_probs don't have same shape "
                                "of (N-1) dims,while N is the rank of samples")
    with tf.control_dependencies([_assert_same_shape]):
        samples = tf.identity(samples)
        log_probs = tf.identity(log_probs)

    input_x = tf.convert_to_tensor(samples, dtype=tf.float32)
    log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
    log_probs = tf.reshape(log_probs, [-1])
    static_x_shape = input_x.get_shape()
    if not static_x_shape[-1:].is_fully_defined():
        raise ValueError(
            'Inputs {} has undefined last dimension.'.format(input_x.name))
    d = int(static_x_shape[-1])

    # define parameters
    with tf.name_scope('planar_flow_parameters'):
        param_bs, param_us, param_ws = [], [], []
        for iter in range(n_iters):
            param_b = tf.Variable(tf.zeros(shape=[1], dtype=tf.float32),
                                  name='param_b_%d' % iter)
            aux_u = tf.Variable(
                tf.random_normal(shape=[d, 1], mean=0, stddev=0.005,
                                 dtype=tf.float32),
                name='aux_u_%d' % iter)
            param_w = tf.Variable(
                tf.random_normal(shape=[d, 1], mean=0, stddev=0.005,
                                 dtype=tf.float32),
                name='para_w_%d' % iter)
            dot_prod = tf.matmul(param_w, aux_u, transpose_a=True)
            param_u = aux_u + param_w / tf.matmul(param_w, param_w,
                                                  transpose_a=True) \
                * (tf.log(tf.exp(dot_prod) + 1) - 1 - dot_prod)
            param_u = tf.transpose(param_u, name='param_u_%d' % iter)
            param_bs.append(param_b)
            param_ws.append(param_w)
            param_us.append(param_u)

    # forward and log_det_jacobian
    z = tf.reshape(input_x, [-1, d])
    for iter in range(n_iters):
        scalar = tf.matmul(param_us[iter], param_ws[iter], name='scalar')
        scalar = tf.reshape(scalar, [])

        # check invertible
        invertible_check = tf.assert_greater_equal(
            scalar, tf.constant(-1.0, dtype=tf.float32),
            message="w'u must be greater or equal to -1")
        with tf.control_dependencies([invertible_check]):
            scalar = tf.identity(scalar)

        param_w = param_ws[iter]
        activation = tf.tanh(
            tf.matmul(z, param_w, name='score') + param_bs[iter],
            name='activation')
        param_u = param_us[iter]

        reduce_act = tf.reduce_sum(activation, axis=-1)
        det_ja = scalar * (
            tf.constant(1.0, dtype=tf.float32) - reduce_act * reduce_act) \
            + tf.constant(1.0, dtype=tf.float32)
        log_probs -= tf.log(det_ja)
        z = z + tf.matmul(activation, param_u, name='update')
    z = tf.reshape(z, tf.shape(input_x))
    log_probs = tf.reshape(log_probs, tf.shape(input_x)[:-1])

    return z, log_probs


def inv_autoregressive_flow(samples, hidden, log_probs, autoregressive_nn,
                            n_iters, update='normal'):
    """
    Perform Inverse Autoregressive Flow (Kingma 2016) along the last axis of
    inputs.

    :param samples: A N-D (N>=2) `float32` Tensor of shape `[..., d]`, and
        inverse autoregressive flow will be performed along the last axis.
    :param hidden: A N-D (N>=2) `float32` Tensor of shape `[..., d]`,
        should be of the same shape as `samples`, whose meaning follows which
        described in (Kingma, 2016).
    :param log_probs: A (N-1)-D `float32` Tensor. should be of the same shape
        as the first N-1 axes of `samples`.
    :param autoregressive_nn: A function, using (name, id, z, hidden) as
        parameters and returning (m, s). See :func:`linear_ar` for an example.
    :param n_iters: A int, which represents the number of successive flows.
    :param update: A string. The update method of flow, if 'normal', will
        use :math:`z = s * z + m`; if 'gru', will use
        :math:`z = \sigma(s) * z + (1 - \sigma(s)) * m`.

    :return: A N-D Tensor, the transformed samples.
    :return: A (N-1)-D Tensor, the log probabilities of the transformed
        samples.
    """
    # TODO: properly deal with hidden.
    samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
    if hidden is not None:
        hidden = tf.convert_to_tensor(hidden, dtype=tf.float32)
    if not isinstance(n_iters, int):
        raise ValueError('n_iters should be type \'int\'')

    # check shapes of samples and log_probs
    static_sample_shape = samples.get_shape()
    static_logprob_shape = log_probs.get_shape()
    static_sample_ndim = convert_to_int(static_sample_shape.ndims)
    static_logprob_ndim = convert_to_int(static_logprob_shape.ndims)
    if static_sample_ndim and static_sample_ndim <= 1:
        raise ValueError('samples should have rank >= 2')
    if static_sample_ndim and static_logprob_ndim \
            and static_sample_ndim != static_logprob_ndim + 1:
        raise ValueError('log_probs should have rank (N-1), while N is the '
                         'rank of samples')
    try:
        tf.broadcast_static_shape(static_sample_shape[:-1],
                                  static_logprob_shape)
    except ValueError:
        raise ValueError(
             "samples and log_probs don't have same shape of (N-1) dims,"
             "while N is the rank of samples")
    dynamic_sample_shape = tf.shape(samples)
    dynamic_logprob_shape = tf.shape(log_probs)
    dynamic_sample_ndim = tf.rank(samples)
    dynamic_logprob_ndim = tf.rank(log_probs)
    _assert_sample_ndim = \
        tf.assert_greater_equal(dynamic_sample_ndim, 2,
                                message='samples should have rank >= 2')
    with tf.control_dependencies([_assert_sample_ndim]):
        samples = tf.identity(samples)
    _assert_logprob_ndim = \
        tf.assert_equal(dynamic_logprob_ndim, dynamic_sample_ndim - 1,
                        message='log_probs should have rank (N-1), while N is'
                                ' the rank of samples')
    with tf.control_dependencies([_assert_logprob_ndim]):
        log_probs = tf.identity(log_probs)
    _assert_same_shape = \
        tf.assert_equal(dynamic_sample_shape[:-1], dynamic_logprob_shape,
                        message="samples and log_probs don't have same shape "
                                "of (N-1) dims,while N is the rank of samples")
    with tf.control_dependencies([_assert_same_shape]):
        samples = tf.identity(samples)
        log_probs = tf.identity(log_probs)

    joint_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
    z = tf.convert_to_tensor(samples, dtype=tf.float32)

    for iter in range(n_iters):
        m, s = autoregressive_nn('iaf', iter, z, hidden)

        if update == 'gru':
            sigma = tf.sigmoid(s)
            z = sigma * z + (1 - sigma) * m
            joint_probs = joint_probs - tf.reduce_sum(tf.log(sigma), axis=-1)

        if update == 'normal':
            z = s * z + m
            joint_probs = joint_probs - tf.reduce_sum(tf.log(s), axis=-1)

        z = tf.reverse(z, [-1])

    return z, joint_probs
