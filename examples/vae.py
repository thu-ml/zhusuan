#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os

import tensorflow as tf
import tflearn
from six.moves import range
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from zhusuan.distributions import norm, bernoulli
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
    v = grad_and_vars[1][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


class M1:
    """
    The deep generative model used in variational autoencoder (VAE).
    """
    def __init__(self):
        pass

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
        l_x_z = tf.reshape(z, (-1, int(z.get_shape()[2])))
        l_x_z = tflearn.fully_connected(l_x_z, 500, activation='relu')
        with tf.device(None):
            l_x_z = tflearn.batch_normalization(l_x_z)
        l_x_z = tflearn.fully_connected(l_x_z, 500, activation='relu')
        with tf.device(None):
            l_x_z = tflearn.batch_normalization(l_x_z)
        l_x_z = tflearn.fully_connected(l_x_z, int(x.get_shape()[1]),
                                        activation='sigmoid')
        l_x_z = tf.reshape(l_x_z, (-1, int(z.get_shape()[1]),
                                   int(x.get_shape()[1])))
        l_x_z = bernoulli.logpdf(tf.expand_dims(x, 1), l_x_z, eps=1e-6)
        log_px_z = tf.reduce_sum(l_x_z, 2)
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
    l_z_x = tflearn.fully_connected(x, 500, activation='relu')
    with tf.device(None):
        l_z_x = tflearn.batch_normalization(l_z_x)
    l_z_x = tflearn.fully_connected(l_z_x, 500, activation='relu')
    with tf.device(None):
        l_z_x = tflearn.batch_normalization(l_z_x)
    l_z_x_mean = tflearn.fully_connected(l_z_x, n_z, activation='linear')
    l_z_x_logstd = tflearn.fully_connected(l_z_x, n_z, activation='linear')
    return l_z_x_mean, l_z_x_logstd


def advi(model, x, vz_mean, vz_logstd, n_samples=1,
         optimizer=tf.train.AdamOptimizer()):
    """
    Implements the automatic differentiation variational inference (ADVI)
    algorithm. For now we assume all latent variables have been transformed in
    the model definition to have support on R^n.

    :param model: An model object that has a method logprob(z, x) to compute
        the log joint likelihood of the model.
    :param x: 2-D Tensor of shape (batch_size, n_x). Observed data.
    :param vz_mean: A Tensorflow node that has shape (batch_size, n_z), which
        denotes the mean of the variational posterior to be optimized during
        the inference.
        For traditional mean-field variational inference, the batch_size can
        be set to 1.
        For amortized variational inference, vz_mean depends on x and should
        have the same batch_size as x.
    :param vz_logstd: A Tensorflow node that has shape (batch_size, n_z), which
        denotes the log standard deviation of the variational posterior to be
        optimized during the inference. See vz_mean for proper usage.
    :param n_samples: Int. Number of posterior samples used to
        estimate the gradients. Default to be 1.
    :param optimizer: Tensorflow optimizer object. Default to be
        AdamOptimizer.

    :return: A Tensorflow computation graph of the inference procedure.
             compute the gradient but dest not apply yet
    :return: A 0-D Tensor. The variational lower bound
    """
    samples = norm.rvs(
        size=(tf.shape(vz_mean)[0], n_samples, tf.shape(vz_mean)[1])) * \
        tf.exp(tf.expand_dims(vz_logstd, 1)) + tf.expand_dims(vz_mean, 1)
    samples.set_shape((None, n_samples, None))
    lower_bound = model.log_prob(samples, x) - tf.reduce_sum(
        norm.logpdf(samples, tf.expand_dims(vz_mean, 1),
                    tf.expand_dims(tf.exp(vz_logstd), 1)), 2)
    lower_bound = tf.reduce_mean(lower_bound)
    #return optimizer.minimize(-lower_bound), lower_bound
    return optimizer.compute_gradients(-lower_bound), lower_bound


if __name__ == "__main__":
    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)

        
        tower_grads = []
        tower_lower_bound = []
        feed_x = []
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('multigpu', i)) as scope:
                    # Run the inference
                    model = M1()
                    x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
                    feed_x.append(x)
                    n_z = 40
                    vz_mean, vz_logstd = q_net(x, n_z)
                    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)
                    #infer, lower_bound = advi(model, x, vz_mean, vz_logstd, 1, optimizer)
                    grad, lower_bound = advi(model, x, vz_mean, vz_logstd, 1, optimizer)
                    #########################################################
                    #I know this call is necessary 
                    #I just follow cifar-10 multi-gpu version
                    #But I don't know its rationale
                    tf.get_variable_scope().reuse_variables() 
                    #########################################################
                    tower_grads.append(grad)
                    tower_lower_bound.append(lower_bound)
                    #init = tf.initialize_all_variables()
        grads = average_gradients(tower_grads)
        apply_gradients_op = optimizer.apply_gradients(grads)

        epoches = 500
        batch_size = 100
        test_batch_size = 200
        #iters = x_train.shape[0] // batch_size
        iters = x_train.shape[0] // (FLAGS.num_gpus * batch_size)
        test_iters = x_test.shape[0] // test_batch_size
        test_freq = 10
        config = tf.ConfigProto(allow_soft_placement=True)
        #with tf.Session() as sess:
        with tf.Session(config=config) as sess:
            #sess.run(init)
            init = tf.initialize_all_variables().run()
            for epoch in range(epoches):
                np.random.shuffle(x_train)
                #tflearn.is_training(True)
                lbs = []
                for t in range(iters):
                    x_batch = x_train[t * (FLAGS.num_gpus * batch_size) :
                                      (t + 1) * (FLAGS.num_gpus * batch_size)]
                    x_batch = np.random.binomial(
                        n=1, p=x_batch, size=x_batch.shape).astype('float32')

                    feed_dict = {}
                    for x_idx, x_data in enumerate(feed_x):
                        feed_dict[x_data] = x_batch[x_idx * batch_size : (x_idx + 1) * batch_size]

                    #_, lb = sess.run([infer, lower_bound], feed_dict={x: x_batch})
                    _, lb = sess.run([apply_gradients_op, lower_bound], feed_dict=feed_dict)
                    lbs.append(lb)
                print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))
                if epoch % test_freq == 0:
                    tflearn.is_training(False)
                    test_lbs = []
                    for t in range(test_iters):
                        test_x_batch = x_test[
                            t * test_batch_size: (t + 1) * test_batch_size]
                        test_lb = sess.run(lower_bound,
                                           feed_dict={x: test_x_batch})
                        test_lbs.append(test_lb)
                    print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
