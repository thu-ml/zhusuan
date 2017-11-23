Variational Autoencoders: Step by Step
======================================

**Variational AutoEncoders** (VAE) :cite:`vae-kingma2013auto` is one of the
most widely used deep generative models. In this tutorial, we show how to
implement VAE in ZhuSuan step by step. The full script is at
`examples/variational_autoencoders/vae.py <https://github.com/thu-ml/zhusuan/blob/master/examples/variational_autoencoders/vae.py>`_.

The generative process of a VAE for modeling binarized
`MNIST <https://www.tensorflow.org/get_started/mnist/beginners>`_ data is as
follows:

.. math::

    z &\sim \mathrm{N}(z|0, I) \\
    x_{logits} &= f_{NN}(z) \\
    x &\sim \mathrm{Bernoulli}(x|\mathrm{sigmoid}(x_{logits}))

This generative process is a stereotype for deep generative models, which
starts with a latent representation (:math:`z`) sampled from a simple
distribution (such as standard Normal). Then the samples are forwarded through
a deep neural network (:math:`f_{NN}`) to capture the complex generative
process of high dimensional observations such as images. Finally, some noise
is added to the output to get a tractable likelihood for the model. For
binarized MNIST, the observation noise is chosen to be Bernoulli, with
its parameters output by the neural network.

Build the Model
---------------

In ZhuSuan, a model is constructed under a
:class:`~zhusuan.model.base.BayesianNet` context, which enables transparent
building of directed graphical models using both Tensorflow operations and
ZhuSuan's :class:`~zhusuan.model.base.StochasticTensor` s.

To start a :class:`~zhusuan.model.base.BayesianNet` context, use "with"
statement in python::

    import zhusuan as zs

    with zs.BayesianNet() as model:
        # Start building the model

Following the generative process, first we need a standard Normal
distribution to generate the latent representations (:math:`z`). As presented
in our graphical model, the data is generated in batches with batch size ``n``,
and for each data, the latent representation is of dimension ``z_dim``. So we
need the Normal to generate samples of shape ``[n, z_dim]``::

    with zs.BayesianNet() as model:
        # z ~ N(z|0, I)
        z_mean = tf.zeros([n, z_dim])
        z = zs.Normal('z', z_mean, std=1., group_ndims=1)

The shape of ``z_mean`` is ``[n, z_dim]``, which means that
we have ``[n, z_dim]`` independent inputs fed into the univariate
:class:`~zhusuan.model.stochastic.Normal` StochasticTensor. Because
input parameters are allowed to
`broadcast <https://docs.scipy.org/doc/numpy-1.12.0/user/basics.broadcasting.html>`_
to match each other's shape, the standard deviation ``std`` is simply set to
1. Thus the shape of samples and probabilities evaluated at this node should
be of shape ``[n, z_dim]``. However, what we want in modeling MNIST data, is a
batch of ``[n]`` independent events, with each one producing samples of ``z``
that is of shape ``[z_dim]``, which is the dimension of latent representations.
And the probabilities in every single event in the batch should be evaluated
together, so the shape of local probabilities should be ``[n]`` instead of
``[n, z_dim]``. In ZhuSuan, the way to achieve this is by setting
``group_ndims``, as we do in the above model definition code. To
understand this, see :ref:`dist-and-stochastic`.

Then we build a neural network of two fully-connected layers with :math:`z` as
the input, which is supposed to learn the complex transformation that
generates images from their latent representations::

    from tensorflow.contrib import layers

    with zs.BayesianNet() as model:
        ...
        # x_logits = f_NN(z)
        lx_z = layers.fully_connected(z, 500)
        lx_z = layers.fully_connected(lx_z, 500)
        x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)

Next, we add an observation distribution (noise) to get a tractable
likelihood when evaluating the probability of an image::

    with zs.BayesianNet() as model:
        ...
        # x ~ Bernoulli(x|sigmoid(x_logits))
        x = zs.Bernoulli('x', x_logits, group_ndims=1)

.. note::

    The :class:`~zhusuan.model.stochastic.Bernoulli` StochasticTensor
    accepts log-odds of probabilities instead of probabilities of being 1.
    This is designed for numeric stability reasons. Similar tricks are used in
    :class:`~zhusuan.model.stochastic.Categorical`, which accepts log
    probabilities instead of probabilities.

Putting together, the code for constructing a VAE is::

    import tensorflow as tf
    from tensorflow.contrib import layers
    import zhusuan as zs

    with zs.BayesianNet() as model:
        z_mean = tf.zeros([n, z_dim])
        z = zs.Normal('z', z_mean, std=1., group_ndims=1)

        lx_z = layers.fully_connected(z, 500)
        lx_z = layers.fully_connected(lx_z, 500)
        x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)

        x = zs.Bernoulli('x', x_logits, group_ndims=1)

Reuse the Model
---------------

Unlike common deep learning models (MLP, CNN, etc.), which is for supervised
tasks, a key difficulty in designing programing primitives for generative
models is their inner reusability. This is because in a probabilistic
graphical model, a stochastic node can have two kinds of
states, **observed or not observed**. Consider the above case, if ``z`` is a
tensor sampled from the prior, how about when you meet the condition that ``z``
is observed? In common practice of tensorflow programming, one has to build
another computation graph from scratch and reuse the Variables (weights here).
If there are many stochastic nodes in the model, this process will be really
painful.

**ZhuSuan has a novel solution for this.** To observe any stochastic nodes,
pass a dictionary mapping of ``(name, Tensor)`` pairs when constructing
:class:`~zhusuan.model.base.BayesianNet`. This will assign observed values
to corresponding ``StochasticTensor`` s. For example, to observe
a batch of images ``x_batch``, write::

    with zs.BayesianNet(observed={'x': x_batch}):
        ...
        x = zs.Bernoulli('x', x_logits, group_ndims=1)

In this case, when ``x`` is used in further computation, it will convert to
the observed value, i.e., ``x_batch``, instead of the sampled tensor.

.. note::

    The observation passed must have the same type and shape as the
    `StochasticTensor`.

..
   With the help of both the ``BayesianNet`` context and factory pattern
   style programing.

To reuse the code above for different observations, a common practice in
ZhuSuan is to wrap it in a function, like this::

    @zs.reuse('model')
    def vae(observed, n, n_x, z_dim):
        with zs.BayesianNet(observed=observed) as model:
            z_mean = tf.zeros([n, z_dim])
            z = zs.Normal('z', z_mean, std=1., group_ndims=1)
            lx_z = layers.fully_connected(z, 500)
            lx_z = layers.fully_connected(lx_z, 500)
            x_logits = layers.fully_connected(lx_z, n_x,
                                              activation_fn=None)
            x = zs.Bernoulli('x', x_logits, group_ndims=1)
        return model

Each time the function is called, a different observation assignment can be
passed. One may ask how to **reuse tensorflow variables** created in this
function. ZhuSuan provides an very easy way to achieve this, that is, without
careful management of variable scopes, one could just add a decorator to the
function: ``@zs.reuse(scope)``, as shown in the above code. Then this function
will automatically create variables the first time they are called and reuse
them thereafter.

Inference and Learning
----------------------

Having built the model, the next step is to learn it from binarized MNIST
images. We conduct
`Maximum Likelihood <https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>`_
learning, that is, we are going to maximize the log likelihood of data in our
model:

.. math::

    \max_{\theta} \log p_{\theta}(x)

where :math:`\theta` is the model parameter.

.. note::

    In this Variational Autoencoder, the model parameter is the network
    weights, in other words, it's the tensorflow variables created in the
    ``fully_connected`` layers.

However, the model we defined has not only the observation (:math:`x`) but
also latent representation (:math:`z`). This makes it hard for us to compute
:math:`p_{\theta}(x)`, which we call the marginal likelihood of :math:`x`,
because we only know the joint likelihood of the model:

.. math::

    p_{\theta}(x, z) = p(z)p_{\theta}(x|z)

while computing the marginal likelihood requires an integral over latent
representation, which is generally intractable:

.. math::

    p_{\theta}(x) = \int p_{\theta}(x, z)\;dz

The intractable integral problem is a fundamental challenge in learning latent
variable models like VAE. Fortunately, the machine learning society has
developed many approximate methods to address it. One of them is
`Variational Inference <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`_.
As the inner intuition is very simple, we briefly introduce it below.

Because directly optimizing :math:`\log p_{\theta}(x)` is infeasible, we choose
to optimize a lower bound of it. The lower bound is constructed as

.. math::

    \log p_{\theta}(x) &\geq \log p_{\theta}(x) - \mathrm{KL}(q_{\phi}(z|x)\|p_{\theta}(z|x)) \\
    &= \mathbb{E}_{q_{\phi}(z|x)} \left[\log p_{\theta}(x, z) - \log q_{\phi}(z|x)\right] \\
    &= \mathcal{L}(\theta, \phi)

where :math:`q_{\phi}(z|x)` is a user-specified distribution of :math:`z`
(called **variational posterior**) that is chosen to match the true posterior
:math:`p_{\theta}(z|x)`. The lower bound is equal to the marginal log
likelihood if and only if :math:`q_{\phi}(z|x) = p_{\theta}(z|x)`, when the
`Kullback–Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_
between them (:math:`\mathrm{KL}(q_{\phi}(z|x)\|p_{\theta}(z|x))`) is zero.

.. note::

    In Bayesian Statistics, the process represented by the Bayes' rule

    .. math::

        p(z|x) = \frac{p(z)(x|z)}{p(x)}

    is called
    `Bayesian Inference <https://en.wikipedia.org/wiki/Bayesian_inference>`_,
    where :math:`p(z)` is called **prior**, :math:`p(x|z)` is the conditional
    likelihood, :math:`p(x)` is the marginal likelihood or **evidence**,
    :math:`p(z|x)` is called **posterior**.

This lower bound is usually called Evidence Lower Bound (ELBO). Note that the
only probabilities we need to evaluate in it is the joint likelihood and
the probability of the variational posterior.

In variational autoencoder, the variational posterior (:math:`q_{\phi}(z|x)`)
is also parameterized by a neural network (:math:`g`), which accepts input
:math:`x`, and outputs the mean and variance of a Normal distribution:

.. math::

    \mu_z(x;\phi), \log\sigma_z(x;\phi) = g_{NN}(x) \\

    q_{\phi}(z|x) = \mathrm{N}(z|\mu_z(x;\phi), \sigma^2_z(x;\phi))

In ZhuSuan, the variational posterior can also be defined as a
:class:`~zhusuan.model.base.BayesianNet`. The code for above definition is::

    @zs.reuse('variational')
    def q_net(x, z_dim):
        with zs.BayesianNet() as variational:
            lz_x = layers.fully_connected(tf.to_float(x), 500)
            lz_x = layers.fully_connected(lz_x, 500)
            z_mean = layers.fully_connected(lz_x, z_dim,
                                            activation_fn=None)
            z_logstd = layers.fully_connected(lz_x, z_dim,
                                              activation_fn=None)
            z = zs.Normal('z', z_mean, logstd=z_logstd, group_ndims=1)
        return variational

There are many ways to optimize this lower bound. One of the easiest way is
to do
`stochastic gradient descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_,
which is very common in deep learning literature. However, the gradient
computation here involves taking derivatives of an expectation, which
needs Monte Carlo estimation. This often induces large variance if not properly
handled.

Many solutions have been proposed to estimate the gradient of some
type of variational lower bound (ELBO or others) with relatively low variance.
To make this more automatic and easier to handle, ZhuSuan has wrapped them
all into :mod:`single functions <zhusuan.variational>`, which computes
the final objective (or surrogate cost) for users to directly take derivatives
on. This means that optimizing these objectives is equally optimizing the
corresponding variational lower bounds using the well-developed low-variance
estimator.

Here we are using the **Stochastic Gradient Variational Bayes** (SGVB)
estimator from the original paper of variational autoencoders
:cite:`vae-kingma2013auto`. This estimator takes benefits of a clever
reparameterization trick to greatly reduce the variance when estimating the
gradients of ELBO. In ZhuSuan, one can use this estimator by calling the method
:func:`~sgvb` of the output of :func:`~zhusuan.variational.elbo`.
The code for this part is::

    x = tf.placeholder(tf.int32, shape=[None, x_dim], name='x')
    n = tf.shape(x)[0]

    def log_joint(observed):
        model = vae(observed, n, x_dim, z_dim)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    variational = q_net(x, z_dim)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    lower_bound = zs.variational.elbo(
        log_joint, observed={'x': x}, latent={'z': [qz_samples, log_qz]})
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

.. note::

    For readers who are interested, we provide a detailed explanation of the
    :func:`~sgvb` estimator used here, though this is not
    required for you to use ZhuSuan's variational functionality.

    The key of SGVB estimator is a reparameterization trick, i.e., they
    reparameterize the random variable
    :math:`z\sim q_{\phi}(z|x) = \mathrm{N}(z|\mu_z(x;\phi), \sigma^2_z(x;\phi))`,
    as

    .. math::

        z = z(\epsilon; x, \phi) = \epsilon \sigma_z(x;\phi) + \mu_z(x;\phi),\; \epsilon\sim \mathrm{N}(0, I)

    In this way, the expectation can be rewritten with respect to
    :math:`\epsilon`:

    .. math::

        \mathcal{L}(\phi, \theta) &=
        \mathbb{E}_{z\sim q_{\phi}(z|x)} \left[\log p_{\theta}(x, z) - \log q_{\phi}(z|x)\right] \\
        &= \mathbb{E}_{\epsilon\sim \mathrm{N}(0, I)} \left[\log p_{\theta}(x, z(\epsilon; x, \phi)) -
        \log q_{\phi}(z(\epsilon; x, \phi)|x)\right]

    Thus the gradients with variational parameters :math:`\phi` can be
    directly exchanged into the expectation, enabling an unbiased low-variance
    Monte Carlo estimator:

    .. math::

        \nabla_{\phi} L(\phi, \theta) &=
        \mathbb{E}_{\epsilon\sim \mathrm{N}(0, I)} \nabla_{\phi} \left[\log p_{\theta}(x, z(\epsilon; x, \phi)) -
        \log q_{\phi}(z(\epsilon; x, \phi)|x)\right] \\
        &\approx \frac{1}{k}\sum_{i=1}^k \nabla_{\phi} \left[\log p_{\theta}(x, z(\epsilon_i; x, \phi)) -
        \log q_{\phi}(z(\epsilon_i; x, \phi)|x)\right]

    where :math:`\epsilon_i \sim \mathrm{N}(0, I)`

Now that we have had the objective function, the next step is to do the
stochastic gradient descent. Tensorflow provides many advanced
`optimizers <https://www.tensorflow.org/api_guides/python/train>`_
that improves the plain SGD, among which Adam
:cite:`vae-kingma2014adam` is probably the most popular one in deep learning
society. Here we are going to use Tensorflow's Adam optimizer to do the
learning::

    optimizer = tf.train.AdamOptimizer(0.001)
    infer_op = optimizer.minimize(cost)

Generate Images
---------------

What we've done above is to define and learn the model. To see how it
performs, we would like to let it generate some images in the learning process.
For the generating process, we remove the observation noise, i.e.,
the ``Bernoulli`` StochasticTensor. We do this by first change the model
function a little to return one more instance,
the direct output of the neural network (``x_logits``)::

    @zs.reuse('model')
    def vae(observed, n, x_dim, z_dim):
        with zs.BayesianNet(observed=observed) as model:
            ...
            x_logits = layers.fully_connected(lx_z, x_dim, activation_fn=None)
            x = zs.Bernoulli('x', x_logits, group_ndims=1)
        # before change: return model
        return model, x_logits

Then we add a sigmoid function to it to get a "mean" image.
This is done by::

    n_gen = 100
    _, x_logits = vae({}, n_gen, x_dim, z_dim)
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])

Run Gradient Descent
--------------------

Now, everything is good before a run. So we could just open the Tensorflow
session, run the training loop, print statistics, and write generated
images to disk. Keep watching them and have fun :)
::

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch})
                lbs.append(lb)

            print('Epoch {}: Lower bound = {}'.format(
                  epoch, np.mean(lbs)))

            if epoch % save_freq == 0:
                images = sess.run(x_gen)
                name = "results/vae/vae.epoch.{}.png".format(epoch)
                save_image_collections(images, name)


.. rubric:: References

.. bibliography:: ../refs.bib
    :style: unsrtalpha
    :labelprefix: VAE
    :keyprefix: vae-
