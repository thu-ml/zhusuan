#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from functools import reduce
import math

from zhusuan.kernels.base import Kernel

__all__ = [
    'CombinationKernel',
    'SumKernel',
    'ProductKernel',
]


class CombinationKernel(Kernel):
    def __init__(self, kernel_num, kernel_list, dtype=tf.float32):
        super(CombinationKernel, self).__init__(kernel_num, dtype)
        self._kernel_list = []
        for kernel_or_kernel_list in kernel_list:
            if isinstance(kernel_or_kernel_list, self.__class__):
                self._kernel_list.extend(kernel_or_kernel_list.kernel_list)
            elif isinstance(kernel_or_kernel_list, Kernel):
                self._kernel_list.append(kernel_or_kernel_list)
            else:
                raise ValueError('Some objects in kernel list are not kernels!')

    @property
    def kernel_list(self):
        return self._kernel_list

    def __call__(self, x, y):
        raise NotImplementedError()

    def Kdiag(self, x):
        raise NotImplementedError()


class SumKernel(CombinationKernel):
    def __init__(self, kernel_num, kernel_list, dtype=tf.float32):
        super(SumKernel, self).__init__(kernel_num, kernel_list, dtype)

    def __call__(self, x, y):
        return reduce(tf.add, [kernel(x, y) for kernel in self._kernel_list])

    def Kdiag(self, x):
        return reduce(tf.add, [kernel.Kdiag(x) for kernel in self._kernel_list])


class ProductKernel(CombinationKernel):
    def __init__(self, kernel_num, kernel_list, dtype=tf.float32):
        super(ProductKernel, self).__init__(kernel_num, kernel_list, dtype)

    def __call__(self, x, y):
        return reduce(tf.multiply, [kernel(x, y) for kernel in self._kernel_list])

    def Kdiag(self, x):
        return reduce(tf.multiply, [kernel.Kdiag(x) for kernel in self._kernel_list])