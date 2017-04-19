Bayesian Neural Network
======================================

Bayesian Neural Network (BNN) is one of the most useful deep neural network models. In this tutorial, we show how to implement
BNN in ZhuSuan step by step. The full script is at
`examples/tutorials/bnn.py <https://github.com/thjashin/ZhuSuan/blob/develop/examples/tutorials/bnn.py>`_.

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
So we need the Normal to generate samples of shape ``[n_out, n_in+1]``::

    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([n_out, n_in + 1])
            w_logstd = tf.zeros([n_out, n_in + 1])
            ws.append(zs.Normal('w' + str(i), w_mu, w_logstd,
                                group_event_ndims=2))

In each layer of ``n_in`` and ``n_out``, the shape of ``w_mu`` and ``w_logstd`` is ``[n_out, n_in+1]``,
which means that we have ``[n_out, n_in+1]`` independent network parameters
fed into the univariate :class:`~zhusuan.model.stochastic.Normal` StochasticTensor.
Thus the shape of samples and probabilities evaluated at this layer should be of
shape ``[n_out, n_in+1]``. However, what we really care is how the data is forwarded throught the neural network. Thus, the probabilities should
be evaluated together, so the shape of local probabilities should be ``[]`` instead
of ``[n_out, n_in+1]``. In ZhuSuan, the way to achieve this is by setting ``group_event_ndims``,
as we do in the above model definition code. To understand this, see
:ref:`dist-and-stochastic`.

Then we write the data forwarding process, through which the connection between output ``y`` and input ``x`` is established::

        # forward
        ly_x = tf.transpose(x, [1, 0])
        for i, w in enumerate(ws):
            ly_x = tf.concat(
                [ly_x, tf.ones([1, tf.shape(x)[0]])], 0)
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.cast(tf.shape(ly_x)[0],
                                                        tf.float32))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(tf.transpose(ly_x, [1, 0]), [1])


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

    @zs.reuse('model')
    def bayesianNN(observed, x, n_x, layer_sizes):
        with zs.BayesianNet(observed=observed) as model:
            ws = []
            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):
                w_mu = tf.zeros([n_out, n_in + 1])
                w_logstd = tf.zeros([n_out, n_in + 1])
                ws.append(zs.Normal('w' + str(i), w_mu, w_logstd,
                                    group_event_ndims=2))
    
            # forward
            ly_x = tf.transpose(x, [1, 0])
            for i, w in enumerate(ws):
                ly_x = tf.concat(
                    [ly_x, tf.ones([1, tf.shape(x)[0]])], 0)
                ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.cast(tf.shape(ly_x)[0],
                                                            tf.float32))
                if i < len(ws) - 1:
                    ly_x = tf.nn.relu(ly_x)
    
            y_mean = tf.squeeze(tf.transpose(ly_x, [1, 0]), [1])
            y_logstd = tf.get_variable('y_logstd', shape=[],
                                       initializer=tf.constant_initializer(0.))
            y = zs.Normal('y', y_mean, y_logstd)
    
        return model, y_mean
    

Inference and Learning
----------------------

Having built the model, the next step is to learn it from the data. With the intractable 
marginal distribution ``p(Y|X)``, we can not directly compute the posterior distribution of netowork parameters (:math:`\{W_i\}_{i=1}^L`).
In order to solve this problem, we use `Variational Inference <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`_,
that is using a variational distribution :math:`q_{\phi}(\{W_i\}_{i=1}^L)=\prod_{i=1}^L{q_{\phi_i}(W_i)}` to approximate the true posterior.
In standard Bayesian Neural Network, the variational posterior (:math:`q_{\phi_i}(W_i)`)
is also parameterized by a Normal distribution parameterized by mean and log standard deviation.

.. math::

    q_{\phi_i}(W_i) = \mathrm{N}(W_i|\mu_i, {\sigma_i}^2)

In ZhuSuan, the variational posterior can also be defined as a
:class:`~zhusuan.model.base.BayesianNet`. The code for above definition is::

    def mean_field_variational(layer_sizes):
        with zs.BayesianNet() as variational:
            ws = []
            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):
                w_mean = tf.get_variable(
                    'w_mean_' + str(i), shape=[n_out, n_in + 1],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable(
                    'w_logstd_' + str(i), shape=[n_out, n_in + 1],
                    initializer=tf.constant_initializer(0.))
                ws.append(
                    zs.Normal('w' + str(i), w_mean, w_logstd,
                              group_event_ndims=2))
        return variational

Following Variational Inference settings, we only need to maximize a lower bound of log marginal pdf (:math:`\log p_{\theta}(Y|X)`):

.. math::

    \log p_{\theta}(Y|X) &\geq \log p_{\theta}(Y|X) - \mathrm{KL}(q_{\phi}(\{W_i\}_{i=1}^L)\|p_{\theta}(\{W_i\}_{i=1}^L)) \\
    &= \mathbb{E}_{q_{\phi}(\{W_i\}_{i=1}^L}) \left[\log p_{\theta}(Y, \{W_i\}_{i=1}^L|X) - \log q_{\phi}(\{W_i\}_{i=1}^L)\right] \\
    &= \mathcal{L}(\theta, \phi)

The lower bound is equal to the marginal log
likelihood if and only if :math:`q_{\phi_i}(W_i) = p_{\theta_i}(W_i)`, for :math:`i` in :math:`1\cdots L`, when the
`Kullback–Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_
between them (:math:`\mathrm{KL}(q_{\phi}(\{W_i\}_{i=1}^L)\|p_{\theta}(\{W_i\}_{i=1}^L))`) is zero.

This lower bound is usually called Evidence Lower Bound (ELBO). Note that the
only probabilities we need to evaluate in it is the joint likelihood and
the probability of the variational posterior.

.. Note::

    Different with some other models like VAE, BNN's parameters :math:`\{W_i\}_{i=1}^L` are global for all the data, therefore the ELBO has a slightly different expression.

We use `stochastic gradient descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_ to optimize this lower bound. The code for this part is::

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])

    def log_joint(observed):
        model, _ = bayesianNN(observed, x, n_x, layer_sizes)
        log_pws = model.local_log_prob(w_names)
        log_py_xw = model.local_log_prob('y')
        return tf.add_n(log_pws) + tf.reduce_mean(log_py_xw) * N

    variational = mean_field_variational(layer_sizes)
    qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
    latent = dict(zip(w_names, qw_outputs))
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'y': y}, latent))

    optimizer = tf.train.AdamOptimizer(0.001, epsilon=1e-4)
    infer = optimizer.minimize(-lower_bound)

Evaluation
---------------

What we've done above is to define and learn the model. To see how it
performs, we would like to compute some quantitative measurements including `Root Mean Squared Error (RMSE) <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ and `log likelihood <https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood>`_.

RMSE is defined as the square root of the predicative mean square error, smaller RMSE means better predictive accuracy:

.. math::
    RMSE = \sqrt{\mathrm{E}(y_{pred}-y)^2}

Log likelihood (ll) is defined as the natural logarithm of the likelihood function,
larger ll means that the learned model fits the data better. In our setting, the output ``y`` has a normal distribution, therefore:

.. math::
    ll = -\frac{(y-\mu)^2}{2\sigma^2} - 0.5\log(2\pi) - \log(\sigma)

First we need to pass the data and sampled latent parameters to the BNN model::

    # prediction: rmse & log likelihood
    observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
    observed.update({'y': y})
    model, y_mean = bayesianNN(observed, x, n_x, layer_sizes)

To be noted, as we have standardized ``y_train`` with ``std_y_train`` to make them
have 1 variance at beginning (check the full script `examples/tutorials/bnn.py <https://github.com/thjashin/ZhuSuan/blob/develop/examples/tutorials/bnn.py>`_), we need to count its effect in our evaluation formulas. 
RMSE is proportional to the amplitude, therefore the final RMSE should be multiplied with ``std_y_train``.
For log likelihood, :math:`\sigma` is proportional to ``y``, therefore the ``std_y_train`` only affects the last term :math:`\log(\sigma)`.
The code is as follows::

    # prediction: rmse & log likelihood
    observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
    observed.update({'y': y})
    model, y_mean = bayesianNN(observed, x, n_x, layer_sizes)
    rmse = tf.sqrt(tf.reduce_mean((y_mean - y) ** 2)) * std_y_train
    log_py_xw = model.local_log_prob('y')
    log_likelihood = tf.reduce_mean(log_py_xw) - tf.log(std_y_train)

.. Note::

    In this illustrating tutorial, we only generate one sample for network parameters in training and evaluation, while multi-sample can have better result especially in evalution. For multi-sample code, please refer to `examples/bayesian_neural_nets/bayesian_nn.py <https://github.com/thjashin/ZhuSuan/blob/develop/examples/bayesian_neural_nets/bayesian_nn.py>`_

Run Gradient Descent
--------------------

Now, everything is good before a run. So we could just open the Tensorflow session, 
run the training loop, and print statistics. Keep watching them and have fun :)::

    # Define training/evaluation parameters
    epoches = 500
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 10

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            indices = np.random.permutation(N)
            x_train = x_train[indices, :]
            y_train = y_train[indices]
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer, lower_bound],
                    feed_dict={x: x_batch, y: y_batch})
                lbs.append(lb)
            print('Epoch {}: Lower bound = {}'.format(
                epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                test_lb, test_rmse, test_ll = sess.run(
                    [lower_bound, rmse, log_likelihood],
                    feed_dict={x: x_test, y: y_test})
                print('>>> TEST')
                print('>> Test lower bound = {}'.format(test_lb))
                print('>> Test rmse = {}'.format(test_rmse))
                print('>> Test log_likelihood = {}'.format(test_ll))
