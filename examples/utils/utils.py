#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import os

import numpy as np
from six.moves import range
from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity


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
