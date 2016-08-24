#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import pytest

from .context import zhusuan
from zhusuan.mcmc.adapters import *


def test_stepsize_adapter():
    # A dummy acceptance rate function, which is monotonically decreasing
    def acceptance_rate_func(x):
        return 1 - 1. / (1 + np.exp(-np.log(x)))

    current_epsilon = 1
    adapter = StepsizeAdapter(m_adapt=100)
    adapter.log_epsilon = np.log(current_epsilon)
    adapter.mu = np.log(10 * current_epsilon)

    for i in range(120):
        acceptance_rate = acceptance_rate_func(current_epsilon)
        current_epsilon = adapter.adapt(acceptance_rate)

    assert(abs(acceptance_rate - 0.8) < 0.05)
