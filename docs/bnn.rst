Bayesian Neural Network: Step by Step
======================================

Bayesian Neural Network (BNN) is one of the most useful deep neural network models. In this tutorial, we show how to implement
BNN in ZhuSuan step by step. The full script is at
`examples/bayesian_nn.py <https://github.com/thjashin/ZhuSuan/blob/develop/examples/bayesian_nn.py>`_.

The forwarding process of a BNN for modeling multivariate regression `boston housing <https://archive.ics.uci.edu/ml/datasets/Housing>`_ data is as follows:

.. math::

    W_i &\sim \mathrm{N}(W_i|0, I),\quad i\quad \mathrm{in}\quad 1 \cdots L. \\
    y_{mean} &= f_{NN}(x, \{W_i\}_{i=1}^L) \\
    y &\sim \mathrm{N}(y|y_{mean}, \sigma^2)

This forwarding process starts with a input (:math:`x`), then the input (:math:`x`)
is forwarded through a deep neural network (:math:`f_{NN}`) whose parameters
in each layer satisfy a multivariate standard Normal distribution. With this complex forwarding
process, the model is enabled to learn complex relationships between the input
(:math:`x`) and the output (:math:`y`). Finally, some noise is added to the
output to get a tractable likelihood for the model, which usually is a Gaussian noise.

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

Following the forwarding process, first we need standard Normal
distributions to generate the latent representations (:math:`\{W_i\}_{i=1}^L`). As presented
in our graphical model, the latent network parameters (:math:`\{W_i\}_{i=1}^L`), of
shape ``[n_out, n_in+1]`` (one gained column for bias), are global for all the data.
So we need the Normal to generate samples of shape ``[1, n_out, n_in+1]``::

    with zs.BayesianNet() as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # W ~ N(W|0, I)
            w_mu = tf.zeros([1, n_out, n_in + 1])
            w_logstd = tf.zeros([1, n_out, n_in + 1])
            ws.append(zs.Normal('w' + str(i), w_mu, w_logstd,
                                n_samples=n_particles, group_event_ndims=2))


In each layer of ``n_in`` and ``n_out``, the shape of ``w_mu`` and ``w_logstd`` is ``[1, n_out, n_in+1]``,
which means that we have ``[1, n_out, n_in+1]`` independent network parameters
fed into the univariate :class:`~zhusuan.model.stochastic.Normal` StochasticTensor.
Thus the shape of samples and probabilities evaluated at this layer should be of
shape ``[1, n_out, n_in+1]``. However, what we really care is how the data is forwarded throught the neural network. Thus, the probabilities in every single event should
be evaluated together, so the shape of local probabilities should be ``[1]`` instead
of ``[1, n_out, n_in+1]``. In ZhuSuan, the way to achieve this is by setting ``group_event_ndims``,
as we do in the above model definition code. To understand this, see
:ref:`dist-and-stochastic`.

Then we write the data forwarding process, through which the connection between output ``y`` and input ``x`` is established::

    with zs.BayesianNet() as model:
        ...
        # have got ws
        ly_x = tf.expand_dims(
            tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        for i in range(len(ws)):
            w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat(
                [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.cast(tf.shape(ly_x)[2],
                                                        tf.float32))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(ly_x, [2, 3])


Next, we add an observation distribution (noise) to get a tractable
likelihood when evaluating the probability::

    with zs.BayesianNet() as model:
        ...
        y_logstd = tf.get_variable('y_logstd', shape=[],
                                   initializer=tf.constant_initializer(0.))
        y = zs.Normal('y', y_mean, y_logstd)


Putting together, the code for constructing a BNN is::

    import tensorflow as tf
    import zhusuan as zs

    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([1, n_out, n_in + 1])
            w_logstd = tf.zeros([1, n_out, n_in + 1])
            ws.append(zs.Normal('w' + str(i), w_mu, w_logstd,
                                n_samples=n_particles, group_event_ndims=2))

        # forward
        ly_x = tf.expand_dims(
            tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        for i in range(len(ws)):
            w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat(
                [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.cast(tf.shape(ly_x)[2],
                                                        tf.float32))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(ly_x, [2, 3])
        y_logstd = tf.get_variable('y_logstd', shape=[],
                                   initializer=tf.constant_initializer(0.))
        y = zs.Normal('y', y_mean, y_logstd)


Reuse the Model
---------------

In order to avoid bothering to build another computation graph and share the network
weights in different scenarios, such as whether :math:`W_i` is observed,
**ZhuSuan proposes a novel solution.** To observe any stochastic nodes,
pass a dictionary mapping of ``(name, Tensor)`` pairs when constructing
:class:`~zhusuan.model.base.BayesianNet`. This will assign observed values
to corresponding ``StochasticTensor`` s. For example, to observe
a batch of outputs ``y_batch``, write::

    with zs.BayesianNet(observed={'y': y_batch}):
        ...
        y = zs.Normal('y', y_mean, y_logstd)

In this case, when ``y`` is used in further computation, it will convert to
the observed value, i.e., ``y_batch``, instead of the sampled tensor.

.. Note::

    The observation passed must have the same type and shape as the
    `StochasticTensor`.

..
   With the help of both the ``BayesianNet`` context and factory pattern
   style programing.

To reuse the code above for different observations, a common practice in
ZhuSuan is to wrap it in a function, like this::
    
    @zs.reuse('model')
    def bayesianNN(observed, x, n_x, layer_sizes, n_particles):
        with zs.BayesianNet(observed=observed) as model:
            ws = []
            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):
                w_mu = tf.zeros([1, n_out, n_in + 1])
                w_logstd = tf.zeros([1, n_out, n_in + 1])
                ws.append(zs.Normal('w' + str(i), w_mu, w_logstd,
                                    n_samples=n_particles, group_event_ndims=2))
    
            # forward
            ly_x = tf.expand_dims(
                tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
            for i in range(len(ws)):
                w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
                ly_x = tf.concat(
                    [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
                ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.cast(tf.shape(ly_x)[2],
                                                            tf.float32))
                if i < len(ws) - 1:
                    ly_x = tf.nn.relu(ly_x)
    
            y_mean = tf.squeeze(ly_x, [2, 3])
            y_logstd = tf.get_variable('y_logstd', shape=[],
                                       initializer=tf.constant_initializer(0.))
            y = zs.Normal('y', y_mean, y_logstd)
    
        return model, y_mean
    

Each time the function is called, a different observation assignment can be
passed. One may ask how to **reuse tensorflow variables** created in this
function. ZhuSuan provides an very easy way to achieve this, that is, without
careful management of variable scopes, one could just add a decorator to the
function: ``@zs.reuse(scope)``, as shown in the above code. Then this function
will automatically create variables the first time they are called and reuse
them thereafter.

Inference and Learning
----------------------

Having built the model, the next step is to learn it from boston housing dataset.
We conduct
`Maximum Likelihood <https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>`_
learning, that is, we are going to maximize the log likelihood of data in our
model:

.. math::

    \max_{\theta} \log p_{\theta}(Y|X)

where :math:`\theta` is the model parameter.

However, the model we defined has not only the observation (:math:`x, y`) but
also latent representation (:math:`\{W_i\}_{i=1}^L`). This makes it hard for us to compute
:math:`p_{\theta}(Y|X)`, which we call the marginal likelihood,
because we only know the joint likelihood of the model:

.. math::

    p_{\theta}(Y, \{W_i\}_{i=1}^L|X) = \prod_{i=1}^L{p_{\theta}(W_i)}p(Y|X, \{W_i\}_{i=1}^L)

while computing the marginal likelihood requires an integral over latent
representation, which is generally intractable:

.. math::

    p_{\theta}(Y|X) = \int p_{\theta}(Y, \{W_i\}_{i=1}^L|X)\;dW_1\cdots dW_L

The intractable integral problem is a fundamental challenge in Bayesian inference problems. Fortunately, the machine learning society has
developed many approximate methods to address it. One of them is
`Variational Inference <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`_.
As the inner intuition is very simple, we briefly introduce it below.

Because directly optimizing :math:`\log p_{\theta}(Y|X)` is infeasible, we choose
to optimize a lower bound of it. The lower bound is constructed as

.. math::

    \log p_{\theta}(Y|X) &\geq \log p_{\theta}(Y|X) - \mathrm{KL}(q_{\phi}(\{W_i\}_{i=1}^L)\|p_{\theta}(\{W_i\}_{i=1}^L)) \\
    &= \mathbb{E}_{q_{\phi}(\{W_i\}_{i=1}^L}) \left[\log p_{\theta}(Y, \{W_i\}_{i=1}^L|X) - \log q_{\phi}(\{W_i\}_{i=1}^L)\right] \\
    &= \mathcal{L}(\theta, \phi)

where :math:`q_{\phi}(\{W_i\}_{i=1}^L)=\prod_{i=1}^L{q_{\phi_i}(W_i)}` is a user-specified distribution of :math:`\{W_i\}_{i=1}^L`
(called **variational posterior**) that is chosen to match the true posterior
:math:`p_{\theta}(\{W_i\}_{i=1}^L)=\prod_{i=1}^L{p_{\theta_i}(W_i)}`. The lower bound is equal to the marginal log
likelihood if and only if :math:`q_{\phi_i}(W_i) = p_{\theta_i}(W_i)`, for :math:`i` in :math:`1\cdots L`, when the
`Kullback–Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_
between them (:math:`\mathrm{KL}(q_{\phi}(\{W_i\}_{i=1}^L)\|p_{\theta}(\{W_i\}_{i=1}^L))`) is zero.

.. Note::

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

In standard Bayesian Neural Network, the variational posterior (:math:`q_{\phi_i}(W_i)`)
is also parameterized by a Normal distribution parameterized by mean and log standard deviation.

.. math::

    q_{\phi_i}(W_i) = \mathrm{N}(W_i|\mu_i, {\sigma_i}^2)

In ZhuSuan, the variational posterior can also be defined as a
:class:`~zhusuan.model.base.BayesianNet`. The code for above definition is::

    def mean_field_variational(layer_sizes, n_particles):
        with zs.BayesianNet() as variational:
            ws = []
            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):
                w_mean = tf.get_variable(
                    'w_mean_' + str(i), shape=[1, n_out, n_in + 1],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable(
                    'w_logstd_' + str(i), shape=[1, n_out, n_in + 1],
                    initializer=tf.constant_initializer(0.))
                ws.append(
                    zs.Normal('w' + str(i), w_mean, w_logstd,
                              n_samples=n_particles, group_event_ndims=2))
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
gradients of ELBO. In ZhuSuan, one can use this estimator by calling the
:func:`~zhusuan.variational.sgvb` function. The code for this part is::

   # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    def log_joint(observed):
        model, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
        log_pws = model.local_log_prob(w_names)
        log_py_xw = model.local_log_prob('y')
        return tf.add_n(log_pws) + tf.reduce_mean(log_py_xw, 1,
                                                  keep_dims=True) * N

    variational = mean_field_variational(layer_sizes, n_particles)
    qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
    latent = dict(zip(w_names, qw_outputs))
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'y': y_obs}, latent, axis=0))

.. Note::

    For readers who are interested, we provide a detailed explanation of the
    :func:`~zhusuan.variational.sgvb` estimator used here, though this is not
    required for you to use ZhuSuan's variational functionality.

    The key of SGVB estimator is a reparameterization trick, i.e., they
    reparameterize the random variable, for global parameters,
    :math:`W_i\sim q_{\phi_i}(W_i) = \mathrm{N}(W_i|\mu_i, {\sigma_i}^2)`,
    as

    .. math::

        W_i = g(\epsilon_i;\; \mu_i, \sigma_i) = \epsilon_i \sigma_i + \mu_i,\; \epsilon_i\sim \mathrm{N}(0, I)

    In this way, the expectation can be rewritten with respect to
    :math:`\epsilon`:

    .. math::

        \mathcal{L}(\phi, \theta) &=
        \mathbb{E}_{W\sim q_{\phi}(W)} \left[\log p_{\theta}(Y, W|X) - \log q_{\phi}(W)\right] \\
        &= \mathbb{E}_{\epsilon\sim \mathrm{N}(0, I)} \left[\log p_{\theta}(Y, \{g(\epsilon_i;\; \mu_i, \sigma_i)\}_1^L|X) -
        \sum\log q(g(\epsilon_i;\; \mu_i, \sigma_i))\right]

    Thus the gradients with variational parameters :math:`\phi` can be
    directly exchanged into the expectation, enabling an unbiased low-variance
    Monte Carlo estimator:

    .. math::

        \nabla_{\phi} L(\phi, \theta) &=
        \mathbb{E}_{\epsilon\sim \mathrm{N}(0, I)} \nabla_{\phi} \left[\log p_{\theta}(Y, \{g(\epsilon_i;\; \mu_i, \sigma_i)\}_1^L|X) -
        \sum\log q(g(\epsilon_i;\; \mu_i, \sigma_i))\right] \\
        &\approx \frac{1}{k}\sum_{i=1}^k \nabla_{\phi} \left[\log p_{\theta}(Y, \{g(\epsilon_i;\; \mu_i, \sigma_i)\}_1^L|X) -
        \sum\log q(g(\epsilon_i;\; \mu_i, \sigma_i))\right]

    where :math:`\epsilon_i \sim \mathrm{N}(0, I)`

Now that we have had the objective function, the next step is to do the
stochastic gradient descent. Tensorflow provides many advanced
`optimizers <https://www.tensorflow.org/api_guides/python/train>`_
that improves the plain SGD, among which Adam
:cite:`vae-kingma2014adam` is probably the most popular one in deep learning
society. Here we are going to use Tensorflow's Adam optimizer to do the
learning::

    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    infer = optimizer.minimize(-lower_bound)

Evaluation
---------------

What we've done above is to define and learn the model. To see how it
performs, we would like to compute some quantitative measurement including `Root Mean Squared Error (RMSE) <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ and `log likelihood <https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood>`_.
First we need to pass the data and sampled latent parameters to the BNN model::
    # prediction: rmse & log likelihood
    observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
    observed.update({'y': y_obs})
    model, y_mean = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
To compute the RMSE and log likelihood, as we generated ``n_particles`` samples
of the latent parameters in the first dimension, we need to average along this dimension, and then compute the RMSE and log likelihood as follows::
    y_pred = tf.reduce_mean(y_mean, 0)
    rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
    log_py_xw = model.local_log_prob('y')
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
        tf.log(std_y_train)

Run Gradient Descent
--------------------

Now, everything is good before a run. So we could just open the Tensorflow session, 
run the training loop, and print statistics. Keep watching them and have fun :)::

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer, lower_bound],
                    feed_dict={n_particles: lb_samples,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lb, test_rmse, test_ll = sess.run(
                    [lower_bound, rmse, log_likelihood],
                    feed_dict={n_particles: ll_samples,
                               x: x_test, y: y_test})
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(test_lb))
                print('>> Test rmse = {}'.format(test_rmse))
                print('>> Test log_likelihood = {}'.format(test_ll))


.. rubric:: References

.. bibliography:: refs.bib
    :style: unsrtalpha
    :keyprefix: vae-
