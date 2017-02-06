#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np


class Hamiltonian:
    """
    A Hamiltonian with diagonal mass matrix.

    :param mass: The diagonal of the mass matrix, which is a list of numpy
    arrays.
    """
    def __init__(self, mass):
        self.mass = mass

    def velocity(self, momentum):
        """
        Returns the velocity according to a given momentum.

        :param momentum: The momentum.
        :return: The velocity.
        """
        return map(lambda(x, y): x/y, zip(momentum, self.mass))

    def energy(self, momentum, potential=0):
        """
        Returns the Hamiltonian (energy) given the momentum and the potential
        energy.

        :param momentum: The momentum.
        :param potential: The potential
        :return: The Hamiltonian.
        """
        kinetic = 0
        for i in range(len(momentum)):
            kinetic += np.sum(momentum[i] * momentum[i] / self.mass[i])

        kinetic *= 0.5
        return kinetic + potential

    def sample(self):
        """
        Sample a new momentum.

        :return: A sample of momentum.
        """
        return map(
            lambda mass: np.random.normal(size=mass.shape) * np.sqrt(mass),
            self.mass)
