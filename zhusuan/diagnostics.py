#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

from six.moves import range
import numpy as np


__all__ = [
    'effective_sample_size',
    'effective_sample_size_1d',
]


def effective_sample_size_1d(samples):
    """
    Compute the effective sample size of a chain of scalar samples.

    :param samples: A 1-D numpy array. The chain of samples.
    :return: A float. The effective sample size.
    """
    n = samples.shape[0]
    mu_hat = np.mean(samples)
    var = np.var(samples) * n / (n - 1)
    var_plus = var * (n - 1) / n

    def auto_covariance(lag):
        return np.mean((samples[:n - lag] - mu_hat) * (samples[lag:] - mu_hat))

    sum_rho = 0
    for t in range(0, n):
        rho = 1 - (var - auto_covariance(t)) / var_plus
        if rho < 0:
            break
        sum_rho += rho

    ess = n / (1 + 2 * sum_rho)
    return ess


def effective_sample_size(samples, burn_in=100):
    """
    Compute the effective sample size of a chain of vector samples, using the
    algorithm in Stan. Users should flatten their samples as vectors if not so.

    :param samples: A 2-D numpy array of shape ``(M, D)``, where ``M`` is the
        number of samples, and ``D`` is the number of dimensions of each
        sample.
    :param burn_in: The number of discarded samples.

    :return: A 1-D numpy array. The effective sample size.
    """
    current_ess = np.inf
    esses = []
    for d in range(samples.shape[1]):
        ess = effective_sample_size_1d(np.squeeze(samples[burn_in:, d]))
        assert ess >= 0
        if ess > 0:
            current_ess = min(current_ess, ess)

        esses.append(ess)
    return current_ess
