#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division


def regular_div(x, y):
    # Tensorflow has deprecated Python 2 division semantics,
    # regular division in Python 3 is true division.
    return x / y


def floor_div(x, y):
    return x // y
