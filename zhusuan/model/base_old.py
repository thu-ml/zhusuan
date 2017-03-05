#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import OrderedDict
from functools import wraps
import re

from six.moves import zip
import tensorflow as tf

from .utils import Context


__all__ = [
    'StochasticTensor',
    'StochasticGraph',
    'reuse',
]


class StochasticTensor(object):
    """
    The :class:`StochasticTensor` class is the base class for various
    distributions used when building stochastic graphs. It is a wrapper
    on Tensor instances which enables transparent building of stochastic graphs
    using Tensorflow primitives.

    :param name: A string. The name of the StochasticTensor. Must be unique in
        the graph.
    :param incomings: A list of Tensors. Parameters needed to specify the
        distribution.
    :param dtype: The type of the resulting Tensor.
    """

    def __init__(self, name, incomings, dtype=None):
        self.name = name
        self.incomings = incomings
        self.dtype = dtype
        self.s_graph = StochasticGraph.get_context()
        self.s_graph._add_stochastic_tensor(self)

    @property
    def tensor(self):
        """
        Return corresponding Tensor through sampling, or if observed, return
        the observed value.

        :return: A Tensor.
        """
        if not hasattr(self, '_tensor'):
            if self.name in self.s_graph.observed:
                try:
                    self._tensor = tf.convert_to_tensor(
                        self.s_graph.observed[self.name], dtype=self.dtype)
                except ValueError as e:
                    raise ValueError("StochasticTensor('{}') not compatible "
                                     "with its observed value. "
                                     "Error message: {}".format(self.name, e))
            else:
                self._tensor = self.sample()
        return self._tensor

    def sample(self, **kwargs):
        """
        Return samples from the distribution.

        :return: A Tensor.
        """
        raise NotImplementedError()

    def log_prob(self, given):
        """
        Compute log probability density (mass) function at `given` value.

        :param given: A Tensor. The value at which to evaluate log probability
            density (mass) function.
        :return: A Tensor. The log probability density (mass) evaluated.
        """
        raise NotImplementedError()

    # overloading arithmetic operations
    def __abs__(self):
        return tf.abs(self)

    def __neg__(self):
        return tf.negative(self)

    def __pos__(self):
        return self

    def __add__(self, other):
        return tf.add(self, other)

    def __radd__(self, other):
        return tf.add(other, self)

    def __sub__(self, other):
        return tf.subtract(self, other)

    def __rsub__(self, other):
        return tf.subtract(other, self)

    def __mul__(self, other):
        return tf.multiply(self, other)

    def __rmul__(self, other):
        return tf.multiply(other, self)

    def __truediv__(self, other):
        return tf.div(self, other)

    __div__ = __truediv__

    def __rtruediv__(self, other):
        return tf.div(other, self)

    __rdiv__ = __rtruediv__

    def __mod__(self, other):
        return tf.mod(self, other)

    def __rmod__(self, other):
        return tf.mod(other, self)

    def __pow__(self, other):
        return tf.pow(self, other)

    def __rpow__(self, other):
        return tf.pow(other, self)

    # logical operations
    def __invert__(self):
        return tf.logical_not(self)

    def __and__(self, other):
        return tf.logical_and(self, other)

    def __or__(self, other):
        return tf.logical_or(self, other)

    def __xor__(self, other):
        return tf.logical_xor(self, other)

    # boolean operations
    def __lt__(self, other):
        return tf.less(self, other)

    def __le__(self, other):
        return tf.less_equal(self, other)

    def __gt__(self, other):
        return tf.greater(self, other)

    def __ge__(self, other):
        return tf.greater_equal(self, other)

    def __eq__(self, other):
        return tf.equal(self, other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.tensor)

    @staticmethod
    def _to_tensor(value, dtype=None, name=None, as_ref=False):
        if dtype and not dtype.is_compatible_with(value.dtype):
            raise ValueError("Incompatible type conversion requested to type "
                             "'{}' for variable of type '{}'".
                             format(dtype.name, value.dtype.name))
        if as_ref:
            raise ValueError("{}: Ref type is not supported.".format(value))
        return value.tensor


tf.register_tensor_conversion_function(
    StochasticTensor, StochasticTensor._to_tensor)


class StochasticGraph(Context):
    """
    A context class supporting model construction in ZhuSuan as stochastic
    graphs.

    :param observed: A dictionary of (string, Tensor) pairs, which maps from
        names of random variables to their observed values.
    """

    def __init__(self, observed=None):
        self.observed = observed if observed else {}
        self.stochastic_tensors = OrderedDict()

    def _add_stochastic_tensor(self, s_tensor):
        """
        Add a StochasticTensor to the graph. This is the function called when
        a StochasticTensor is created in the context.

        :param s_tensor: A :class:`StochasticTensor` instance.
        """
        if s_tensor.name in self.stochastic_tensors:
            raise ValueError("There has been a StochasticTensor with name "
                             "'{}' in the graph. Names should be unique".
                             format(s_tensor.name))
        else:
            self.stochastic_tensors[s_tensor.name] = s_tensor

    def outputs(self, name_or_names):
        """
        Get the outputs of StochasticTensors by their names, through running
        generative process on the graph. For observed variables, their observed
        values are returned; while for latent variables, samples are returned.

        :param name_or_names: A string or a list of strings. Names of
            StochasticTensors in the graph.
        :return: A Tensor or a list of Tensors.
        """
        if isinstance(name_or_names, (tuple, list)):
            return [self.stochastic_tensors[name].tensor
                    for name in name_or_names]
        else:
            return self.stochastic_tensors[name_or_names].tensor

    def local_log_prob(self, name_or_names):
        """
        Get local probability density (mass) values of StochasticTensors by
        their names. For observed variables, the probability is evaluated at
        their observed values; for latent variables, the probability is
        evaluated at their sampled values.

        :param name_or_names: A string or a list of strings. Names of
            StochasticTensors in the graph.
        :return: A Tensor or a list of Tensors.
        """
        if isinstance(name_or_names, (tuple, list)):
            ret = []
            for name in name_or_names:
                s_tensor = self.stochastic_tensors[name]
                ret.append(s_tensor.log_prob(s_tensor.tensor))
        else:
            s_tensor = self.stochastic_tensors[name_or_names]
            ret = s_tensor.log_prob(s_tensor.tensor)
        return ret

    def query(self, name_or_names, outputs=False, local_log_prob=False):
        """
        Make probabilistic queries on the StochasticGraph. Various options
        are available:
        * outputs: See `StochasticGraph.outputs()`.
        * local_log_prob: See `StochasticGraph.local_log_prob()`.
        For each queried StochasticTensor, a tuple containing results of
        selected options is returned.

        :param name_or_names: A string or a list of strings. Names of
            StochasticTensors in the graph.
        :param outputs: A bool. Whether to query outputs.
        :param local_log_prob: A bool. Whether to query local log probability
            density (mass) values.

        :return: Tuple of Tensors or a list of tuples of Tensors.
        """
        ret = []
        if outputs:
            ret.append(self.outputs(name_or_names))
        if local_log_prob:
            ret.append(self.local_log_prob(name_or_names))
        if len(ret) == 0:
            raise ValueError("No query options are selected.")
        elif isinstance(name_or_names, (tuple, list)):
            return list(zip(*ret))
        else:
            return tuple(ret)


def reuse(scope):
    """
    A decorator for transparent reuse of `tf.Variable` s in a function.
    When a `StochasticGraph` is reused as in a function, this decorator helps
    reuse the `tf.Variable` s in the graph every time the function is called.

    :param scope: A string. The scope name passed to `tf.variable_scope()`.
    """

    def reuse_decorator(f):
        @wraps(f)
        def _func(*args, **kwargs):
            try:
                with tf.variable_scope(scope, reuse=True):
                    return f(*args, **kwargs)
            except ValueError as e:
                if re.search(r'.*not exist.*tf\.get_variable.*', str(e)):
                    with tf.variable_scope(scope):
                        return f(*args, **kwargs)
                else:
                    raise

        return _func

    return reuse_decorator
