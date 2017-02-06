#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
from zhusuan.utils import copy


class Candidate:
    """
    The :class:`Candidate` class represents a random item drawn from a
    collection of items. It supports adding an item to the collection and
    merging two collections, while maintaining the random sample. Either
    `shape` or `init` need to be provided.

    :param shape: The shape of each item.
    :param init: An initial item.
    """
    def __init__(self, shape=None, init=None):
        if int(shape is None) + int(init is None) != 1:
            raise RuntimeError('Only one of shape and init should be '
                               'specified')

        if shape is not None:
            self.shape = shape
            self.reset()
        else:
            self.shape = map(lambda x: x.shape, init)
            self.count = 1
            self.x = copy(init)

    def reset(self):
        """
        Clear the collection.
        """
        self.x = map(lambda shape: np.zeros(shape), self.shape)
        self.count = 0

    def add(self, y):
        """
        Add a new item to the collection, and maintain the sample.
        :param y: The new item.
        """
        self.count += 1
        if np.random.random() < 1. / self.count:
            self.x = copy(y)

    def merge(self, candidate):
        """
        Merge two collections.
        :param candidate: Another collection.
        """
        # TODO I think this is incorrect, but NUTS paper and Stan use this...
        if np.random.random() < float(candidate.count) / self.count:
            self.x = copy(candidate.x)
        self.count += candidate.count
