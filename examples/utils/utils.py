#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import os

import numpy as np
from six.moves import range
from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity
from tensorflow.contrib.framework.python.ops import add_arg_scope
import tensorflow as tf
from tensorflow.contrib import layers


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def save_image_collections(x, filename, shape=(10, 10), scale_each=False,
                           transpose=False):
    """
    :param shape: tuple
        The shape of final big images.
    :param x: numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param scale_each: bool
        If true, rescale intensity for each image.
    :param transpose: bool
        If true, transpose x to (number_of_images, rows, columns, channels),
        i.e., put channels behind.
    :return: `uint8` numpy array
        The output image.
    """
    makedirs(filename)
    n = x.shape[0]
    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


def average_rmse_over_batches(rmses, sizes):
    """
    Average rmses over batches (may not be of the same size).

    :param rmses: A list of per-batch rmses.
    :param sizes: A list of batch sizes.
    :return: The average rmse.
    """
    rmses = np.array(rmses)
    sizes = np.array(sizes)
    return np.sqrt(np.sum(rmses ** 2 * sizes) / np.sum(sizes))


@add_arg_scope
def conv2d_transpose(
        inputs,
        out_shape,
        kernel_size=(5, 5),
        stride=(1, 1),
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=layers.xavier_initializer(),
        scope=None,
        reuse=None):
    batchsize = tf.shape(inputs)[0]
    in_channels = int(inputs.get_shape()[-1])

    output_shape = tf.stack([batchsize, out_shape[0],
                             out_shape[1], out_shape[2]])
    filter_shape = [kernel_size[0], kernel_size[1], out_shape[2], in_channels]

    with tf.variable_scope(scope, 'Conv2d_transpose', [inputs], reuse=reuse):
        w = tf.get_variable('weights', filter_shape,
                            initializer=weights_initializer)

        outputs = tf.nn.conv2d_transpose(
            inputs, w, output_shape=output_shape,
            strides=[1, stride[0], stride[1], 1])
        outputs.set_shape([None] + out_shape)

        if not normalizer_fn:
            biases = tf.get_variable('biases', [out_shape[2]],
                                     initializer=tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, biases)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs
