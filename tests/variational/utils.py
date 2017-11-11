#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def _kl_normal_normal(mean1=0., std1=1., mean2=0., std2=1.):
    return tf.log(std2 / std1) + (std1**2 + (mean1 - mean2)**2) / \
        (2 * std2**2) - 0.5
