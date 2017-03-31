Variational Autoencoders: Step by Step
======================================

Variational AutoEncoders (VAE) :cite:`vae-kingma2013auto` is one of the most
widely used deep generative models.

The generative process of a VAE for modeling binarized MNIST data is as
follows:

.. math::

    z &\sim \mathrm{N}(z|0, I) \\
    x_{logits} &= f_{NN}(z) \\
    x &\sim \mathrm{Bernoulli}(x|\mathrm{sigmoid}(x_{logits}))

This generative process is a stereotype for deep generative models, which
starts with a latent representation sampled from a simple distribution
(such as standard Normal). Then the samples are forwarded through a deep neural
network (:math:`f_{NN}`) to capture the complex generative process of high
dimensional observations such as images. Finally, some noise is added
to the observation to get a tractable likelihood for the model. For binarized
MNIST, the observation noise is chosen to be Bernoulli, with the
its parameters output by the neural network.

Build the Model
---------------

In ZhuSuan, a model is constructed under a
:class:`~zhusuan.model.base.BayesianNet` context, which enables transparent
building of directed graphical models using both Tensorflow operations and
ZhuSuan's :class:`~zhusuan.model.base.StochasticTensor` s. The code for
constructing a VAE is like:
::

    import tensorflow as tf
    from tensorflow.contrib import layers
    import zhusuan as zs

    with zs.BayesianNet() as model:
        # z ~ N(z|0, I)
        z_mean = tf.zeros([n, n_z])
        z_logstd = tf.zeros([n, n_z])
        z = zs.Normal('z', z_mean, z_logstd, group_event_ndims=1)

        # x_logits = f_NN(z)
        lx_z = layers.fully_connected(z, 500)
        lx_z = layers.fully_connected(lx_z, 500)
        x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)

        # x ~ Bernoulli(x|sigmoid(x_logits))
        x = zs.Bernoulli('x', x_logits, group_event_ndims=1)


.. Note::

    The :class:`~zhusuan.model.stochastic.Bernoulli` `StochasticTensor`
    accepts log-odds of probabilities instead of probabilities. This is
    designed for numeric stability reasons. Similar tricks are used in
    :class:`~zhusuan.model.stochastic.Categorical`, which accepts log
    probabilities instead of probabilities.

The shape of ``z_mean`` and ``z_logstd`` is ``[n, n_z]``, which means that
we have ``[n, n_z]`` independent inputs fed into the univariate
:class:`~zhusuan.model.stochastic.Normal` `StochasticTensor`. Thus the shape
of samples and probabilities evaluated at this node should be of shape
``[n, n_z]``. However, what we want in modeling MNIST data, is a batch of
``[n]`` independent events, with each one producing samples of z that is of
shape ``[n_z]``, which is the dimension of latent representations. And the
probabilities in every single event in the batch should be evaluated together,
so the shape of local probabilities should be ``[n]`` instead of ``[n, n_z]``.
In ZhuSuan, the way to achieve this is by setting ``group_event_ndims``,
as we do in the above model definition code. To understand this, see
:ref:`dist-and-stochastic`.

.. rubric:: References

.. bibliography:: refs.bib
    :style: unsrtalpha
    :keyprefix: vae-
