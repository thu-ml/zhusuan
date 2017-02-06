#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import pytest

from tests.context import zhusuan
from zhusuan.mcmc_old.candidate import *


def test_candidate():
    try:
        a = Candidate(init=1, shape=1)
    except RuntimeError:
        pass
