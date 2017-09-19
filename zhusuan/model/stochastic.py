#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from zhusuan import distributions
from zhusuan.model.base import StochasticTensor


__all__ = [
    'Normal',
    'HalfNormal',
    'Bernoulli',
    'Categorical',
    'OnehotCategorical',
    'Discrete',
    'OnehotDiscrete',
    'Uniform',
    'Gamma',
    'Beta',
    'Poisson',
    'Binomial',
    'InverseGamma',
    'Laplace',
    'Multinomial',
    'Dirichlet',
    'BinConcrete',
    'ExpConcrete',
    'Concrete',
]


class Normal(StochasticTensor):
    """
    The class of univariate Normal `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    .. warning::

         The order of arguments `logstd`/`std` will change to `std`/`logstd`
         in the coming version.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param mean: A `float` Tensor. The mean of the Normal distribution.
        Should be broadcastable to match `logstd`.
    :param logstd: A `float` Tensor. The log standard deviation of the Normal
        distribution. Should be broadcastable to match `mean`.
    :param std: A `float` Tensor. The standard deviation of the Normal
        distribution. Should be positive and broadcastable to match `mean`.
    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        `StochasticTensor` are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 mean=0.,
                 logstd=None,
                 std=None,
                 n_samples=None,
                 group_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False,
                 **kwargs):
        norm = distributions.Normal(
            mean,
            logstd=logstd,
            std=std,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        super(Normal, self).__init__(name, norm, n_samples)

#TODO: changed
class HalfNormal(StochasticTensor):
    """
    The class of univariate HalfNormal distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    .. warning::

         The order of arguments `logstd`/`std` will change to `std`/`logstd`
         in the coming version.

    :param mean: A `float` Tensor. The mean of the HalfNormal distribution.
        The mean value must be 0.
        Should be broadcastable to match `logstd`.
    :param logstd: A `float` Tensor. The log standard deviation of the HalfNormal
        distribution. Should be broadcastable to match `mean`.
    :param std: A `float` Tensor. The standard deviation of the HalfNormal
        distribution. Should be positive and broadcastable to match `mean`.
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 mean=0.,
                 logstd=None,
                 std=None,
                 n_samples=None,
                 group_event_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False):
        norm = distributions.HalfNormal(
            mean,
            logstd=logstd,
            std=std,
            group_event_ndims=group_event_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics
        )
        super(HalfNormal, self).__init__(name, norm, n_samples)

class Bernoulli(StochasticTensor):
    """
    The class of univariate Bernoulli `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param logits: A `float` Tensor. The log-odds of probabilities of being 1.

        .. math:: \\mathrm{logits} = \\log \\frac{p}{1 - p}

    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param dtype: The value type of this `StochasticTensor`.
    """

    def __init__(self,
                 name,
                 logits,
                 n_samples=None,
                 group_ndims=0,
                 dtype=None,
                 **kwargs):
        bernoulli = distributions.Bernoulli(
            logits,
            group_ndims=group_ndims,
            dtype=dtype,
            **kwargs
        )
        super(Bernoulli, self).__init__(name, bernoulli, n_samples)


class Categorical(StochasticTensor):
    """
    The class of univariate Categorical `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j,..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param dtype: The value type of this `StochasticTensor`.

    A single sample is a (N-1)-D Tensor with `tf.int32` values in range
    [0, n_categories).
    """

    def __init__(self,
                 name,
                 logits,
                 n_samples=None,
                 group_ndims=0,
                 dtype=None,
                 **kwargs):
        cat = distributions.Categorical(
            logits,
            group_ndims=group_ndims,
            dtype=dtype,
            **kwargs
        )
        super(Categorical, self).__init__(name, cat, n_samples)


Discrete = Categorical


class Uniform(StochasticTensor):
    """
    The class of univariate Uniform `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param minval: A `float` Tensor. The lower bound on the range of the
        uniform distribution. Should be broadcastable to match `maxval`.
    :param maxval: A `float` Tensor. The upper bound on the range of the
        uniform distribution. Should be element-wise bigger than `minval`.
    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        `StochasticTensor` are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 minval=0.,
                 maxval=1.,
                 n_samples=None,
                 group_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False,
                 **kwargs):
        uniform = distributions.Uniform(
            minval,
            maxval,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        super(Uniform, self).__init__(name, uniform, n_samples)


class Gamma(StochasticTensor):
    """
    The class of univariate Gamma `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param alpha: A `float` Tensor. The shape parameter of the Gamma
        distribution. Should be positive and broadcastable to match `beta`.
    :param beta: A `float` Tensor. The inverse scale parameter of the Gamma
        distribution. Should be positive and broadcastable to match `alpha`.
    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 alpha,
                 beta,
                 n_samples=None,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs):
        gamma = distributions.Gamma(
            alpha,
            beta,
            group_ndims=group_ndims,
            check_numerics=check_numerics,
            **kwargs
        )
        super(Gamma, self).__init__(name, gamma, n_samples)


class Beta(StochasticTensor):
    """
    The class of univariate Beta `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param alpha: A `float` Tensor. One of the two shape parameters of the
        Beta distribution. Should be positive and broadcastable to match
        `beta`.
    :param beta: A `float` Tensor. One of the two shape parameters of the
        Beta distribution. Should be positive and broadcastable to match
        `alpha`.
    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 alpha,
                 beta,
                 n_samples=None,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs):
        beta = distributions.Beta(
            alpha,
            beta,
            group_ndims=group_ndims,
            check_numerics=check_numerics,
            **kwargs
        )
        super(Beta, self).__init__(name, beta, n_samples)


class Poisson(StochasticTensor):
    """
    The class of univariate Poisson `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param rate: A `float` Tensor. The rate parameter of Poisson
        distribution. Must be positive.
    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param dtype: The value type of this `StochasticTensor`.
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 rate,
                 n_samples=None,
                 group_ndims=0,
                 dtype=None,
                 check_numerics=False,
                 **kwargs):
        poisson = distributions.Poisson(
            rate,
            group_ndims=group_ndims,
            dtype=dtype,
            check_numerics=check_numerics,
            **kwargs
        )
        super(Poisson, self).__init__(name, poisson, n_samples)


class Binomial(StochasticTensor):
    """
    The class of univariate Binomial `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param logits: A `float` Tensor. The log-odds of probabilities.

        .. math:: \\mathrm{logits} = \\log \\frac{p}{1 - p}

    :param n_experiments: A 0-D `int32` Tensor. The number of experiments
        for each sample.
    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param dtype: The value type of this `StochasticTensor`.
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 logits,
                 n_experiments,
                 n_samples=None,
                 group_ndims=0,
                 dtype=None,
                 check_numerics=False,
                 **kwargs):
        binomial = distributions.Binomial(
            logits,
            n_experiments,
            group_ndims=group_ndims,
            dtype=dtype,
            check_numerics=check_numerics,
            **kwargs
        )
        super(Binomial, self).__init__(name, binomial, n_samples)


class Multinomial(StochasticTensor):
    """
    The class of Multinomial `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param n_experiments: A 0-D `int32` Tensor. The number of experiments
        for each sample.
    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param dtype: The value type of this `StochasticTensor`.

    A single sample is a N-D Tensor with the same shape as logits. Each slice
    `[i, j, ..., k, :]` is a vector of counts for all categories.
    """

    def __init__(self,
                 name,
                 logits,
                 n_experiments,
                 n_samples=None,
                 group_ndims=0,
                 dtype=None,
                 **kwargs):
        multinomial = distributions.Multinomial(
            logits,
            n_experiments,
            group_ndims=group_ndims,
            dtype=dtype,
            **kwargs
        )
        super(Multinomial, self).__init__(name, multinomial, n_samples)


class OnehotCategorical(StochasticTensor):
    """
    The class of one-hot Categorical `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param dtype: The value type of this `StochasticTensor`.

    A single sample is a N-D Tensor with the same shape as logits. Each slice
    `[i, j, ..., k, :]` is a one-hot vector of the selected category.
    """

    def __init__(self,
                 name,
                 logits,
                 n_samples=None,
                 group_ndims=0,
                 dtype=None,
                 **kwargs):
        onehot_cat = distributions.OnehotCategorical(
            logits,
            group_ndims=group_ndims,
            dtype=dtype,
            **kwargs
        )
        super(OnehotCategorical, self).__init__(name, onehot_cat, n_samples)


OnehotDiscrete = OnehotCategorical


class Dirichlet(StochasticTensor):
    """
    The class of Dirichlet `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param alpha: A N-D (N >= 1) `float` Tensor of shape (..., n_categories).
        Each slice `[i, j, ..., k, :]` represents the concentration parameter
        of a Dirichlet distribution. Should be positive.
    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param check_numerics: Bool. Whether to check numeric issues.

    A single sample is a N-D Tensor with the same shape as alpha. Each slice
    `[i, j, ..., k, :]` of the sample is a vector of probabilities of a
    Categorical distribution `[x_1, x_2, ... ]`, which lies on the simplex

    .. math:: \\sum_{i} x_i = 1, 0 < x_i < 1

    """

    def __init__(self,
                 name,
                 alpha,
                 n_samples=None,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs):
        dirichlet = distributions.Dirichlet(
            alpha,
            group_ndims=group_ndims,
            check_numerics=check_numerics,
            **kwargs
        )
        super(Dirichlet, self).__init__(name, dirichlet, n_samples)


class InverseGamma(StochasticTensor):
    """
    The class of univariate InverseGamma `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param alpha: A `float` Tensor. The shape parameter of the InverseGamma
        distribution. Should be positive and broadcastable to match `beta`.
    :param beta: A `float` Tensor. The scale parameter of the InverseGamma
        distribution. Should be positive and broadcastable to match `alpha`.
    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 alpha,
                 beta,
                 n_samples=None,
                 group_ndims=0,
                 check_numerics=False,
                 **kwargs):
        inv_gamma = distributions.InverseGamma(
            alpha,
            beta,
            group_ndims=group_ndims,
            check_numerics=check_numerics,
            **kwargs
        )
        super(InverseGamma, self).__init__(name, inv_gamma, n_samples)


class Laplace(StochasticTensor):
    """
    The class of univariate Laplace `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param loc: A `float` Tensor. The location parameter of the Laplace
        distribution. Should be broadcastable to match `scale`.
    :param scale: A `float` Tensor. The scale parameter of the Laplace
        distribution. Should be positive and broadcastable to match `loc`.
    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        `StochasticTensor` are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 loc,
                 scale,
                 n_samples=None,
                 group_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False,
                 **kwargs):
        laplace = distributions.Laplace(
            loc,
            scale,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        super(Laplace, self).__init__(name, laplace, n_samples)


class BinConcrete(StochasticTensor):
    """
    The class of univariate BinConcrete `StochasticTensor` from
    (Maddison, 2016). It is the binary case of :class:`~Concrete`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    .. seealso::

        :class:`~Concrete` and :class:`~ExpConcrete`

    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param temperature: A 0-D `float` Tensor. The temperature of the relaxed
        distribution. The temperature should be positive.
    :param logits: A `float` Tensor. The log-odds of probabilities of being 1.

        .. math:: \\mathrm{logits} = \\log \\frac{p}{1 - p}

    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        `StochasticTensor` are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 temperature,
                 logits,
                 n_samples=None,
                 group_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False,
                 **kwargs):
        bin_concrete = distributions.BinConcrete(
            temperature,
            logits,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        super(BinConcrete, self).__init__(name, bin_concrete, n_samples)


class ExpConcrete(StochasticTensor):
    """
    The class of ExpConcrete `StochasticTensor` from (Maddison, 2016),
    transformed from :class:`~Concrete` by taking logarithm.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    .. seealso::

        :class:`~BinConcrete` and :class:`~Concrete`

    :param temperature: A 0-D `float` Tensor. The temperature of the relaxed
        distribution. The temperature should be positive.
    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        `StochasticTensor` are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 temperature,
                 logits,
                 n_samples=None,
                 group_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False,
                 **kwargs):
        exp_concrete = distributions.ExpConcrete(
            temperature,
            logits,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        super(ExpConcrete, self).__init__(name, exp_concrete, n_samples)


class Concrete(StochasticTensor):
    """
    The class of Concrete `StochasticTensor` from (Maddison, 2016), served as
    the continuous relaxation of the :class:`~OnehotCategorical`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    .. seealso::

        :class:`~BinConcrete` and :class:`~ExpConcrete`

    :param temperature: A 0-D `float` Tensor. The temperature of the relaxed
        distribution. The temperature should be positive.
    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param n_samples: A 0-D `int32` Tensor or None. Number of samples
        generated by this `StochasticTensor`.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        `StochasticTensor` are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 name,
                 temperature,
                 logits,
                 n_samples=None,
                 group_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False,
                 **kwargs):
        concrete = distributions.Concrete(
            temperature,
            logits,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        super(Concrete, self).__init__(name, concrete, n_samples)
