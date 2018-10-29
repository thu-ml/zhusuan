#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = [
    'Kernel',
]


class Kernel(object):
    def __init__(self, kernel_num, dtype=tf.float32):
        self._kernel_num = kernel_num
        self._dtype = dtype

    def __call__(self, x, y):
        raise NotImplementedError()

    def Kdiag(self, x):
        raise NotImplementedError()

    @property
    def kernel_num(self):
        return self._kernel_num

    @property
    def dtype(self):
        return self._dtype