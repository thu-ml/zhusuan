#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np


def ESS1(samples):
    """
    Compute the effective sample size of a chain of scalar samples.

    :param samples: The chain of samples, which is a one dimensional numpy
    array.
    :return: The effective sample size.
    """
    n = samples.shape[0]
    mu_hat = np.mean(samples)
    var = np.var(samples) * n / (n-1)
    var_plus = var * (n-1) / n

    def autocovariance(lag):
        return np.mean((samples[:n-lag]-mu_hat) * (samples[lag:]-mu_hat))

    sum_rho = 0
    for t in range(0, n):
        rho = 1 - (var - autocovariance(t)) / var_plus
        if rho < 0:
            break
        sum_rho += rho

    ess = n / (1 + 2*sum_rho)
    return ess


def ESS(samples, burnin=100):
    """
    Compute the effective sample size of a chain of vector samples, using the
    algorithm in Stan. Users should flatten their samples as vectors if not so.

    :param samples: A M by D matrix, where M is the number of samples, and D is
     the number of dimensions.
    :param burnin: The number of discarded samples.
    """
    current_ess = 1e9
    esss = []
    for d in range(samples.shape[1]):
        ess = ESS1(np.squeeze(samples[burnin:, d]))
        assert(ess >= 0)
        if ess > 0:
            current_ess = min(current_ess, ess)

        esss.append(ess)
    return current_ess
