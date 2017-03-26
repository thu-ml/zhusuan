#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from scipy import misc
import numpy as np

from .context import zhusuan
from zhusuan.transform import *

class TestPlanarNf(tf.test.TestCase):
    def test_planar_nf(self):
    	with self.test_session(use_gpu=True) as sess:

    		for iter in range(10):
				z = []
				vz = [0.1, -1.2, 1.0, -0.3, 1.2, 2, 10.0, -23.2]
				for i in range(len(vz)):
					z.append(np.array([[ vz[i] ]]))
					z[i] = tf.constant(z[i], dtype=tf.float32)
				z_0 = tf.concat(z, axis=1)
				z_1, n_log_det_ja = planar_nf(z_0, tf.constant([0.0],dtype=tf.float32), iters=iter+1)

				n_log_det_ja = tf.reshape(n_log_det_ja, [])

				grad = []
				for i in range(len(vz)):
					z_1i = z_1[0,i]
					grad.append(tf.gradients(z_1i, z_0)[0])
				jocabian = tf.concat(grad, axis=0)
				log_det_jacobian = tf.log(tf.matrix_determinant(jocabian))

				sess.run(tf.global_variables_initializer())
				test_value, true_value = sess.run([-log_det_jacobian, n_log_det_ja])
				self.assertAllClose(test_value, true_value)


class TestLinearIaf(tf.test.TestCase):
    def test_linear_iaf(self):
    	with self.test_session(use_gpu=True) as sess:

			z = []
			vz = [0.1, -1.2, 1.0, -0.3, 1.2, 2, 10.0, -23.2]
			for i in range(len(vz)):
				z.append(np.array([[ vz[i] ]]))
				z[i] = tf.constant(z[i], dtype=tf.float32)
			z_0 = tf.concat(z, axis=1)
			z_1, n_log_det_ja = iaf(z_0, None, tf.constant([0.0],dtype=tf.float32), linear_ar, iters=1)

			n_log_det_ja = tf.reshape(n_log_det_ja, [])

			grad = []
			for i in range(len(vz)):
				z_1i = z_1[0,i]
				grad.append(tf.gradients(z_1i, z_0)[0])
			jocabian = tf.concat(grad, axis=0)
			log_det_jacobian = tf.log(tf.matrix_determinant(jocabian))

			sess.run(tf.global_variables_initializer())
			test_value, true_value = sess.run([-log_det_jacobian, n_log_det_ja])
			self.assertAllClose(test_value, true_value)

