# -*- coding: utf-8 -*-

import tensorflow as tf
from zhusuan.mcmc.hmc import HMC
#from matplotlib import pyplot as plt
import scipy
import scipy.stats


tf.set_random_seed(0)

x = tf.Variable(tf.zeros(shape=[]))

def log_posterior(x):
    return -0.5 * x[0] * x[0]

sampler = HMC(step_size=0.1, num_leapfrog_steps=10)
#init_step = sampler.init_step_size(log_posterior, [x])
sample_step, p_step, new_hamiltonian_step, old_hamiltonian_step, t_step = sampler.sample(log_posterior, [x])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
#step_size = sess.run(init_step)
#print('Step size = {}'.format(step_size))

samples = []
for i in range(10):
    sample, p, new_hamiltonian, old_hamiltonian, t = sess.run([sample_step, p_step,
                                                               new_hamiltonian_step, old_hamiltonian_step, t_step])
    print(sample, p, new_hamiltonian, old_hamiltonian, t)
    #samples.append(sample)
    #print sample, p, new_hamiltonian, old_hamiltonian

#plt.hist(samples, bins=30)
#plt.show()

#print(scipy.stats.normaltest(samples))