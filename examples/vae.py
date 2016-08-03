#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os

import tensorflow as tf
import prettytensor as pt
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
    from zhusuan.utils import log_mean_exp
    from zhusuan.variational import ReparameterizedNormal, advi
except:
    raise ImportError()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dataset
except:
    raise ImportError()

tf.app.flags.DEFINE_integer('num_gpus', 2, """How many GPUs to use""")
FLAGS = tf.app.flags.FLAGS

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    print (grad_and_vars)
    grads = []
    for g, _ in grad_and_vars:
      if g == None: #for no-gradient variable(added by xsz)
        continue
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)
    if len(grads) == 0:
      continue

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).

    :param n_z: Int. The dimension of latent variables (z).
    :param n_x: Int. The dimension of observed variables (x).
    """
    def __init__(self, n_z, n_x):
        self.n_z = n_z
        self.n_x = n_x
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               scale_after_normalization=True):
            self.l_x_z = (pt.template('z').
                          fully_connected(500).
                          batch_normalize().
                          fully_connected(500).
                          batch_normalize().
                          fully_connected(n_x, activation_fn=tf.nn.sigmoid))

    def log_prob(self, z, x):
        """
        The joint likelihood of M1 deep generative model.

        :param z: Tensor of shape (batch_size, samples, n_z). n_z is the
            dimension of latent variables.
        :param x: Tensor of shape (batch_size, n_x). n_x is the dimension of
            observed variables (data).

        :return: A Tensor of shape (batch_size, samples). The joint log
            likelihoods.
        """
        l_x_z = self.l_x_z.construct(
            z=tf.reshape(z, ((-1, self.n_z)))).reshape(
            (-1, int(z.get_shape()[1]), self.n_x)).tensor
        log_px_z = tf.reduce_sum(
            bernoulli.logpdf(tf.expand_dims(x, 1), l_x_z, eps=1e-6), 2)
        log_pz = tf.reduce_sum(norm.logpdf(z), 2)
        return log_px_z + log_pz


def q_net(x, n_z):
    """
    Build the recognition network (Q-net) used as variational posterior.

    :param x: Tensor of shape (batch_size, n_x).
    :param n_z: Int. The dimension of latent variables (z).

    :return: A Tensor of shape (batch_size, n_z). Variational mean of latent
        variables.
    :return: A Tensor of shape (batch_size, n_z). Variational log standard
        deviation of latent variables.
    """
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                           scale_after_normalization=True):
        l_z_x = (pt.wrap(x).
                 fully_connected(500).
                 batch_normalize().
                 fully_connected(500).
                 batch_normalize())
        l_z_x_mean = l_z_x.fully_connected(n_z, activation_fn=None)
        l_z_x_logstd = l_z_x.fully_connected(n_z, activation_fn=None)
    return l_z_x_mean, l_z_x_logstd


def is_loglikelihood(model, x, z_proposal, n_samples=1000):
    """
    Data log likelihood (:math:`\log p(x)`) estimates using self-normalized
    importance sampling.

    :param model: A model object that has a method logprob(z, x) to compute the
        log joint likelihood of the model.
    :param x: A Tensor of shape (batch_size, n_x). The observed variables (
        data).
    :param z_proposal: A :class:`Variational` object used as the proposal
        in importance sampling.
    :param n_samples: Int. Number of samples used in this estimate.

    :return: A Tensor of shape (batch_size,). The log likelihood of data (x).
    """
    samples = z_proposal.sample(n_samples)
    log_w = model.log_prob(samples, x) - z_proposal.logpdf(samples)
    return log_mean_exp(log_w, 1)


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')

    # Define hyper-parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 5000
    epoches = 3000
    batch_size = 100
    test_batch_size = 100
    iters = x_train.shape[0] // batch_size
    test_iters = x_test.shape[0] // test_batch_size
    test_freq = 10

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)
        
        tower_grads = []
        tower_lower_bound = []
        feed_x = []

        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('multigpu', i)) as scope:
                    # Build the training computation graph
                    x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
                    feed_x.append(x)
                    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)
                    with tf.variable_scope("model_%d" %i) as scope:
                        #tf.get_variable_scope().reuse_variables() 
                        with pt.defaults_scope(phase=pt.Phase.train):
                            train_model = M1(n_z, x_train.shape[1])
                            train_vz_mean, train_vz_logstd = q_net(x, n_z)
                            train_variational = ReparameterizedNormal(
                                train_vz_mean, train_vz_logstd)
                            grads, lower_bound = advi(
                                train_model, x, train_variational, lb_samples, optimizer)
                            #infer = optimizer.apply_gradients(grads)

                            tower_grads.append(grads)
                            tower_lower_bound.append(lower_bound)

                    # Build the evaluation computation graph
                    with tf.variable_scope("model_%d" % i, reuse=True) as scope:
                        #tf.get_variable_scope().reuse_variables() 
                        with pt.defaults_scope(phase=pt.Phase.test):
                            eval_model = M1(n_z, x_train.shape[1])
                            eval_vz_mean, eval_vz_logstd = q_net(x, n_z)
                            eval_variational = ReparameterizedNormal(
                                eval_vz_mean, eval_vz_logstd)
                            eval_lower_bound = is_loglikelihood(
                                eval_model, x, eval_variational, lb_samples)
                            eval_log_likelihood = is_loglikelihood(
                                eval_model, x, eval_variational, ll_samples)

        grads_multigpu = average_gradients(tower_grads)
        apply_gradients_op = optimizer.apply_gradients(grads_multigpu)

        init = tf.initialize_all_variables()
        # Run the inference
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
        #with tf.Session() as sess:
            sess.run(init)
            for epoch in range(1, epoches + 1):
                np.random.shuffle(x_train)
                lbs = []
                for t in range(iters):
                    x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                    x_batch = np.random.binomial(
                        n=1, p=x_batch, size=x_batch.shape).astype('float32')
                    feed_dict = {}
                    for x_idx, x_data in enumerate(feed_x):
                        feed_dict[x_data] = x_batch[x_idx * batch_size : (x_idx + 1) * batch_size]
                    #_, lb = sess.run([infer, lower_bound], feed_dict={x: x_batch})
                    _, lb  = sess.run([apply_gradients_op, tower_lower_bound[0]], feed_dict=feed_dict)
                    lbs.append(lb)
                print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))
                #if epoch % test_freq == 0:
                    #test_lbs = []
                    #test_lls = []
                    #for t in range(test_iters):
                        #test_x_batch = x_test[
                            #t * test_batch_size: (t + 1) * test_batch_size]
                        #test_lb, test_ll = sess.run(
                            #[eval_lower_bound, eval_log_likelihood],
                            #feed_dict={x: test_x_batch}
                        #)
                        #test_lbs.append(test_lb)
                        #test_lls.append(test_ll)
                    #print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                    #print('>> Test log likelihood = {}'.format(np.mean(test_lls)))
