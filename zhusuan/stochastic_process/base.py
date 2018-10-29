#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    'StochasticProcess',
]


class StochasticProcess(object):
    def __init__(self):
        pass

    def instantiate(self, positions):
        raise NotImplementedError

    def conditional(self, x, inducing_points, inducing_values):
        raise NotImplementedError

