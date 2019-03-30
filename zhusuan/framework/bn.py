#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.client.session import (
    register_session_run_conversion_functions)
import six

from zhusuan import distributions
from zhusuan.utils import TensorArithmeticMixin
from zhusuan.framework.meta_bn import Local, MetaBayesianNet
from zhusuan.framework.utils import Context


__all__ = [
    'StochasticTensor',
    'BayesianNet',
]


# TODO: __str__, __repr__ for StochasticTensor

class StochasticTensor(TensorArithmeticMixin):
    def __init__(self, bn, name, dist, observation=None, **kwargs):
        if bn is None:
            try:
                bn = BayesianNet.get_context()
            except RuntimeError:
                pass
            else:
                bn.nodes[name] = self

        self._bn = bn
        self._name = name
        self._dist = dist
        self._dtype = dist.dtype
        self._n_samples = kwargs.get("n_samples", None)
        if observation is not None:
            self._observation = self._check_observation(observation)
        elif (self._bn is not None) and (self._name in self._bn._observed):
            self._observation = self._check_observation(
                self._bn._observed[name])
        else:
            self._observation = None
        super(StochasticTensor, self).__init__()

    def _check_observation(self, observation):
        type_msg = "Incompatible types of {}('{}') and its observation: {}"
        try:
            observation = tf.convert_to_tensor(observation, dtype=self._dtype)
        except ValueError as e:
            raise type(e)(
                type_msg.format(self.__class__.__name__, self._name, e))

        shape_msg = "Incompatible shapes of {}('{}') and its observation: " \
                    "{} vs {}."
        dist_shape = self._dist.get_batch_shape().concatenate(
            self._dist.get_value_shape())
        try:
            tf.broadcast_static_shape(dist_shape, observation.get_shape())
        except ValueError as e:
            raise type(e)(
                shape_msg.format(
                    self.__class__.__name__, self._name, dist_shape,
                    observation.get_shape()))
        return observation

    @property
    def bn(self):
        return self._bn

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def dist(self):
        return self._dist

    def is_observed(self):
        return self._observation is not None

    @property
    def tensor(self):
        if self._observation is not None:
            return self._observation
        elif not hasattr(self, "_samples"):
            self._samples = self._dist.sample(n_samples=self._n_samples)
        return self._samples

    @property
    def shape(self):
        """
        Return the static shape of `self.tensor`.

        :return: A `TensorShape` instance.
        """
        return self.tensor.shape

    def get_shape(self):
        """
        Alias of :attr:`shape`.

        :return: A `TensorShape` instance.
        """
        return self.shape

    @property
    def cond_log_p(self):
        if not hasattr(self, "_cond_log_p"):
            self._cond_log_p = self._dist.log_prob(self.tensor)
        return self._cond_log_p

    @staticmethod
    def _to_tensor(value, dtype=None, name=None, as_ref=False):
        if dtype and not dtype.is_compatible_with(value.dtype):
            raise ValueError("Incompatible type conversion requested to type "
                             "'{}' for variable of type '{}'".
                             format(dtype.name, value.dtype.name))
        if as_ref:
            raise ValueError("{}: Ref type not supported.".format(value))
        return value.tensor

    # Below are deprecated features:

    @property
    def net(self):
        return self._bn

    @property
    def distribution(self):
        return self._dist

    def sample(self, n_samples):
        return self._dist.sample(n_samples)

    def log_prob(self, given):
        return self._dist.log_prob(given)

    def prob(self, given):
        return self._dist.prob(given)


tf.register_tensor_conversion_function(
    StochasticTensor, StochasticTensor._to_tensor)

# bring support for session.run(StochasticTensor), and for using as keys
# in feed_dict.
register_session_run_conversion_functions(
    StochasticTensor,
    fetch_function=lambda t: ([t.tensor], lambda val: val[0]),
    feed_function=lambda t, v: [(t.tensor, v)],
    feed_function_for_partial_run=lambda t: [t.tensor]
)


class _BayesianNet(object):
    def __init__(self):
        self._nodes = {}
        try:
            self._local_cxt = Local.get_context()
        except RuntimeError:
            self._local_cxt = None
        if self._local_cxt:
            self._meta_bn = self._local_cxt.meta_bn
        else:
            self._meta_bn = None
        super(_BayesianNet, self).__init__()

    @property
    def nodes(self):
        return self._nodes

    def _get_observation(self, name):
        if self._local_cxt:
            ret = self._local_cxt.observations.get(name, None)
            return ret
        return None

    def stochastic(self, name, dist, **kwargs):
        if name in self._nodes:
            raise ValueError(
                "There exists a node with name '{}' in the {}. Names should "
                "be unique.".format(name, BayesianNet.__name__))
        # TODO: check whether `self` is BayesianNet or _BayesianNet
        node = StochasticTensor(
            self, name, dist, observation=self._get_observation(name), **kwargs)
        self._nodes[name] = node
        return node

    def input(self, name):
        input_tensor = tf.convert_to_tensor(self._get_observation(name))
        self._nodes[name] = input_tensor
        return input_tensor

    def output(self, name, input_tensor):
        self._nodes[name] = input_tensor
        return input_tensor

    # TODO: Deprecate deterministic?
    def deterministic(self, name, input_tensor):
        input_tensor = tf.convert_to_tensor(input_tensor)
        self._nodes[name] = input_tensor
        return input_tensor

    def _check_name_exist(self, name, only_stochastic=False):
        if not isinstance(name, six.string_types):
            raise TypeError(
                "Expected string in `name_or_names`, got {} of type {}."
                .format(repr(name), type(name)))
        if name not in self._nodes:
            raise ValueError("There isn't a node named '{}' in the {}."
                             .format(name, BayesianNet.__name__))
        elif only_stochastic and not isinstance(
                self._nodes[name], StochasticTensor):
            raise ValueError("Node '{}' is deterministic (input or output).".format(name))
        return name

    def _check_names_exist(self, name_or_names, only_stochastic=False):
        """
        Check if there are ``StochasticTensor`` s with `name_or_names` in the
        net.

        :param name_or_names: A string or a tuple(list) of strings. Names of
            ``StochasticTensor`` s in the net.
        :param only_stochastic: A bool. Whether to check only in stochastic
            nodes. Default is `False`.

        :return: The validated name, or a tuple of the validated names.
        """
        if isinstance(name_or_names, six.string_types):
            names = (name_or_names,)
        else:
            name_or_names = tuple(name_or_names)
            names = name_or_names
        for name in names:
            _ = self._check_name_exist(name, only_stochastic=only_stochastic)
        return name_or_names

    def get(self, name_or_names):
        name_or_names = self._check_names_exist(name_or_names)
        if isinstance(name_or_names, tuple):
            return [self._nodes[name] for name in name_or_names]
        else:
            return self._nodes[name_or_names]

    def cond_log_prob(self, name_or_names):
        name_or_names = self._check_names_exist(name_or_names,
                                                only_stochastic=True)
        if isinstance(name_or_names, tuple):
            return [self._nodes[name].cond_log_p for name in name_or_names]
        else:
            return self._nodes[name_or_names].cond_log_p

    def _log_joint(self):
        if (self._meta_bn is None) or (self._meta_bn.log_joint is None):
            ret = sum(node.cond_log_p for node in six.itervalues(self._nodes)
                      if isinstance(node, StochasticTensor))
        elif callable(self._meta_bn.log_joint):
            ret = self._meta_bn.log_joint(self)
        else:
            raise TypeError(
                "{}.log_joint is set to a non-callable instance: {}"
                .format(self._meta_bn.__class__.__name__,
                        repr(self._meta_bn.log_joint)))
        return ret

    def log_joint(self):
        if not hasattr(self, "_log_joint_cache"):
            self._log_joint_cache = self._log_joint()
        return self._log_joint_cache

    def __getitem__(self, name):
        name = self._check_name_exist(name)
        return self._nodes[name]

    def __setitem__(self, name, node):
        raise TypeError(
            "{} instance does not support replacement of the existing node. "
            "To achieve this, pass observations of certain nodes when "
            "calling {}.{}".format(
                BayesianNet.__name__, MetaBayesianNet.__name__,
                MetaBayesianNet.observe.__name__))


class BayesianNet(_BayesianNet, Context):
    def __init__(self, observed=None):
        # To support deprecated features
        self._observed = observed if observed else {}
        super(BayesianNet, self).__init__()

    def normal(self,
               name,
               mean=0.,
               _sentinel=None,
               std=None,
               logstd=None,
               group_ndims=0,
               n_samples=None,
               is_reparameterized=True,
               check_numerics=False,
               **kwargs):
        dist = distributions.Normal(
            mean,
            _sentinel=_sentinel,
            std=std,
            logstd=logstd,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def fold_normal(self,
                    name,
                    mean=0.,
                    _sentinel=None,
                    std=None,
                    logstd=None,
                    n_samples=None,
                    group_ndims=0,
                    is_reparameterized=True,
                    check_numerics=False,
                    **kwargs):
        dist = distributions.FoldNormal(
            mean,
            _sentinel=_sentinel,
            std=std,
            logstd=logstd,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def bernoulli(self,
                  name,
                  logits,
                  n_samples=None,
                  group_ndims=0,
                  dtype=tf.int32,
                  **kwargs):
        dist = distributions.Bernoulli(
            logits,
            group_ndims=group_ndims,
            dtype=dtype,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def categorical(self,
                    name,
                    logits,
                    n_samples=None,
                    group_ndims=0,
                    dtype=tf.int32,
                    **kwargs):
        dist = distributions.Categorical(
            logits,
            group_ndims=group_ndims,
            dtype=dtype,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    discrete = categorical

    def uniform(self,
                name,
                minval=0.,
                maxval=1.,
                n_samples=None,
                group_ndims=0,
                is_reparameterized=True,
                check_numerics=False,
                **kwargs):
        dist = distributions.Uniform(
            minval,
            maxval,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def gamma(self,
              name,
              alpha,
              beta,
              n_samples=None,
              group_ndims=0,
              check_numerics=False,
              **kwargs):
        dist = distributions.Gamma(
            alpha,
            beta,
            group_ndims=group_ndims,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def beta(self,
             name,
             alpha,
             beta,
             n_samples=None,
             group_ndims=0,
             check_numerics=False,
             **kwargs):
        dist = distributions.Beta(
            alpha,
            beta,
            group_ndims=group_ndims,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def poisson(self,
                name,
                rate,
                n_samples=None,
                group_ndims=0,
                dtype=tf.int32,
                check_numerics=False,
                **kwargs):
        dist = distributions.Poisson(
            rate,
            group_ndims=group_ndims,
            dtype=dtype,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def binomial(self,
                 name,
                 logits,
                 n_experiments,
                 n_samples=None,
                 group_ndims=0,
                 dtype=tf.int32,
                 check_numerics=False,
                 **kwargs):
        dist = distributions.Binomial(
            logits,
            n_experiments,
            group_ndims=group_ndims,
            dtype=dtype,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def multivariate_normal_cholesky(self,
                                     name,
                                     mean,
                                     cov_tril,
                                     n_samples=None,
                                     group_ndims=0,
                                     is_reparameterized=True,
                                     check_numerics=False,
                                     **kwargs):
        dist = distributions.MultivariateNormalCholesky(
            mean,
            cov_tril,
            group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def matrix_variate_normal_cholesky(self,
                                       name,
                                       mean,
                                       u_tril,
                                       v_tril,
                                       n_samples=None,
                                       group_ndims=0,
                                       is_reparameterized=True,
                                       check_numerics=False,
                                       **kwargs):
        dist = distributions.MatrixVariateNormalCholesky(
            mean,
            u_tril,
            v_tril,
            group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def multinomial(self,
                    name,
                    logits,
                    n_experiments,
                    normalize_logits=True,
                    n_samples=None,
                    group_ndims=0,
                    dtype=tf.int32,
                    **kwargs):
        dist = distributions.Multinomial(
            logits,
            n_experiments,
            normalize_logits=normalize_logits,
            group_ndims=group_ndims,
            dtype=dtype,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def unnormalized_multinomial(self,
                                 name,
                                 logits,
                                 normalize_logits=True,
                                 group_ndims=0,
                                 dtype=tf.int32,
                                 **kwargs):
        dist = distributions.UnnormalizedMultinomial(
            logits,
            normalize_logits=normalize_logits,
            group_ndims=group_ndims,
            dtype=dtype,
            **kwargs
        )
        return self.stochastic(name, dist, **kwargs)

    bag_of_categoricals = unnormalized_multinomial

    def onehot_categorical(self,
                           name,
                           logits,
                           n_samples=None,
                           group_ndims=0,
                           dtype=tf.int32,
                           **kwargs):
        dist = distributions.OnehotCategorical(
            logits,
            group_ndims=group_ndims,
            dtype=dtype,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    onehot_discrete = onehot_categorical

    def dirichlet(self,
                  name,
                  alpha,
                  n_samples=None,
                  group_ndims=0,
                  check_numerics=False,
                  **kwargs):
        dist = distributions.Dirichlet(
            alpha,
            group_ndims=group_ndims,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def inverse_gamma(self,
                      name,
                      alpha,
                      beta,
                      n_samples=None,
                      group_ndims=0,
                      check_numerics=False,
                      **kwargs):
        dist = distributions.InverseGamma(
            alpha,
            beta,
            group_ndims=group_ndims,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def laplace(self,
                name,
                loc,
                scale,
                n_samples=None,
                group_ndims=0,
                is_reparameterized=True,
                check_numerics=False,
                **kwargs):
        dist = distributions.Laplace(
            loc,
            scale,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    def bin_concrete(self,
                     name,
                     temperature,
                     logits,
                     n_samples=None,
                     group_ndims=0,
                     is_reparameterized=True,
                     check_numerics=False,
                     **kwargs):
        dist = distributions.BinConcrete(
            temperature,
            logits,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    bin_gumbel_softmax = bin_concrete

    def exp_concrete(self,
                     name,
                     temperature,
                     logits,
                     n_samples=None,
                     group_ndims=0,
                     is_reparameterized=True,
                     check_numerics=False,
                     **kwargs):
        dist = distributions.ExpConcrete(
            temperature,
            logits,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    exp_gumbel_softmax = exp_concrete

    def concrete(self,
                 name,
                 temperature,
                 logits,
                 n_samples=None,
                 group_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False,
                 **kwargs):
        dist = distributions.Concrete(
            temperature,
            logits,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
            **kwargs
        )
        return self.stochastic(name, dist, n_samples=n_samples, **kwargs)

    gumbel_softmax = concrete

    # Below are deprecated features:

    def outputs(self, name_or_names):
        name_or_names = self._check_names_exist(name_or_names)
        if isinstance(name_or_names, tuple):
            return [self._nodes[name].tensor for name in name_or_names]
        else:
            return self._nodes[name_or_names].tensor

    def local_log_prob(self, name_or_names):
        return self.cond_log_prob(name_or_names)

    def query(self, name_or_names, outputs=False, local_log_prob=False):
        name_or_names = self._check_names_exist(name_or_names)
        ret = []
        if outputs:
            ret.append(self.outputs(name_or_names))
        if local_log_prob:
            ret.append(self.local_log_prob(name_or_names))
        if len(ret) == 0:
            raise ValueError("No query options are selected.")
        elif isinstance(name_or_names, tuple):
            return list(zip(*ret))
        else:
            return tuple(ret)
