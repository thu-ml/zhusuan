#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import pytest
from mock import Mock
from six.moves import range, map
import prettytensor as pt

from .context import zhusuan
from zhusuan.layers import *


class TestLayer:
    def test_init(self):
        m = Mock()
        layer1 = Layer(m, name='layer1')
        assert(layer1.name == 'layer1')
        assert(layer1.input_layer == m)

    def test_get_output_for(self):
        with pytest.raises(NotImplementedError):
            Layer(Mock()).get_output_for(Mock())


class TestMergeLayer:
    def test_init(self):
        m = [Mock() for _ in range(3)]
        layer1 = MergeLayer(m, name='layer1')
        assert(layer1.name == 'layer1')
        assert(layer1.input_layers == m)

    def test_get_output_for(self):
        with pytest.raises(NotImplementedError):
            MergeLayer([Mock(), Mock()]).get_output_for(Mock())


class TestInputLayer:
    def test_init(self):
        layer1 = InputLayer((None, 500), name='layer1')
        assert(layer1.name == 'layer1')
        assert(layer1.shape == (None, 500))

        with pytest.raises(ValueError):
            InputLayer(shape=(-1,))
        with pytest.raises(ValueError):
            InputLayer((tf.constant(1),))

        data = tf.ones((2, 500))
        layer2 = InputLayer((None, 500), input=data)
        assert(layer2.input == data)

        with pytest.raises(ValueError):
            InputLayer((None, 1, 500), input=np.ones((2, 500)))


class TestReparameterizedNormalOld:
    def test_init(self):
        with pytest.raises(TypeError):
            ReparameterizedNormalOld(Mock())
        with pytest.raises(ValueError):
            ReparameterizedNormalOld([Mock(), ])
        with pytest.raises(ValueError):
            ReparameterizedNormalOld([Mock() for _ in range(3)])

    def test_get_output_for(self):
        mean_ = tf.placeholder(tf.float32, shape=(None, 1, 20))
        logstd = tf.placeholder(tf.float32, shape=(None, 1, 20))
        layer = ReparameterizedNormalOld([Mock(), Mock()], n_samples=10)
        output = layer.get_output_for([mean_, logstd])
        assert(output.get_shape().as_list() == [None, 10, 20])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={mean_: np.random.random((2, 1, 20)),
                                   logstd: np.random.random((2, 1, 20))})
            assert(test_values.shape == (2, 10, 20))

        mean_ = tf.placeholder(tf.float32, shape=(None, 5, 20))
        logstd = tf.placeholder(tf.float32, shape=(None, 5, 20))
        layer = ReparameterizedNormalOld([Mock(), Mock()])
        output = layer.get_output_for([mean_, logstd])
        assert(output.get_shape().as_list() == [None, 5, 20])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={mean_: np.random.random((2, 5, 20)),
                                   logstd: np.random.random((2, 5, 20))})
            assert(test_values.shape == (2, 5, 20))

    def test_get_output_for_n_samples_dynamic(self):
        mean_ = tf.placeholder(tf.float32, shape=(None, 1, 20))
        logstd = tf.placeholder(tf.float32, shape=(None, 1, 20))
        n_samples = tf.placeholder(tf.float32, shape=())
        layer = ReparameterizedNormalOld([Mock(), Mock()], n_samples)
        output = layer.get_output_for([mean_, logstd])
        assert(output.get_shape().as_list() == [None, None, 20])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={mean_: np.random.random((2, 1, 20)),
                                   logstd: np.random.random((2, 1, 20)),
                                   n_samples: 10})
            assert(test_values.shape == (2, 10, 20))

    def test_get_logpdf_for(self):
        mean_ = tf.placeholder(tf.float32, shape=(None, 1, 20))
        logstd = tf.placeholder(tf.float32, shape=(None, 1, 20))
        x = tf.placeholder(tf.float32, shape=(None, 10, 20))
        layer = ReparameterizedNormalOld([Mock(), Mock()], n_samples=10)
        logpdf = layer.get_logpdf_for(x, [mean_, logstd])
        assert(logpdf.get_shape().as_list() == [None, 10])
        with tf.Session() as sess:
            test_values = sess.run(
                logpdf, feed_dict={mean_: np.random.random((2, 1, 20)),
                                   logstd: np.random.random((2, 1, 20)),
                                   x: np.random.random((2, 10, 20)) * 2})
            # TODO: test values
            assert(test_values.shape == (2, 10))


class TestNormal:
    def test_init(self):
        with pytest.raises(TypeError):
            Normal(Mock())
        with pytest.raises(ValueError):
            Normal([Mock(), ])
        with pytest.raises(ValueError):
            Normal([Mock() for _ in range(3)])

    def test_get_output_for(self):
        mean_ = tf.placeholder(tf.float32, shape=(None, 1, 20))
        logvar = tf.placeholder(tf.float32, shape=(None, 1, 20))
        layer = Normal([Mock(), Mock()], n_samples=10)
        output = layer.get_output_for([mean_, logvar])
        assert(output.get_shape().as_list() == [None, 10, 20])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={mean_: np.random.random((2, 1, 20)),
                                   logvar: np.random.random((2, 1, 20))})
            assert(test_values.shape == (2, 10, 20))
        output_d = layer.get_output_for([mean_, logvar], deterministic=True)
        assert(output_d == mean_)

        mean_ = tf.placeholder(tf.float32, shape=(None, 5, 20))
        logvar = tf.placeholder(tf.float32, shape=(None, 5, 20))
        layer = Normal([Mock(), Mock()])
        output = layer.get_output_for([mean_, logvar])
        assert(output.get_shape().as_list() == [None, 5, 20])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={mean_: np.random.random((2, 5, 20)),
                                   logvar: np.random.random((2, 5, 20))})
            assert(test_values.shape == (2, 5, 20))

        mean_ = tf.placeholder(tf.float32, shape=(None, 5, 20, 15))
        logvar = tf.placeholder(tf.float32, shape=(None, 5, 20, 15))
        layer = Normal([Mock(), Mock()])
        output = layer.get_output_for([mean_, logvar])
        assert(output.get_shape().as_list() == [None, 5, 20, 15])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={mean_: np.random.random((2, 5, 20, 15)),
                                   logvar: np.random.random((2, 5, 20, 15))})
            assert(test_values.shape == (2, 5, 20, 15))

    def test_get_output_for_n_samples_dynamic(self):
        mean_ = tf.placeholder(tf.float32, shape=(None, 1, 20))
        logvar = tf.placeholder(tf.float32, shape=(None, 1, 20))
        n_samples = tf.placeholder(tf.float32, shape=())
        layer = Normal([Mock(), Mock()], n_samples)
        output = layer.get_output_for([mean_, logvar])
        assert(output.get_shape().as_list() == [None, None, 20])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={mean_: np.random.random((2, 1, 20)),
                                   logvar: np.random.random((2, 1, 20)),
                                   n_samples: 10})
            assert(test_values.shape == (2, 10, 20))

        mean_ = tf.placeholder(tf.float32, shape=(None, 1, 20, 15))
        logvar = tf.placeholder(tf.float32, shape=(None, 1, 20, 15))
        n_samples = tf.placeholder(tf.float32, shape=())
        layer = Normal([Mock(), Mock()], n_samples)
        output = layer.get_output_for([mean_, logvar])
        assert(output.get_shape().as_list() == [None, None, 20, 15])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={mean_: np.random.random((2, 1, 20, 15)),
                                   logvar: np.random.random((2, 1, 20, 15)),
                                   n_samples: 10})
            assert(test_values.shape == (2, 10, 20, 15))

    def test_get_output_for_reparameterized(self):
        mean_ = tf.placeholder(tf.float32, shape=(None, 1, 2))
        logvar = tf.placeholder(tf.float32, shape=(None, 1, 2))
        layer = Normal([Mock(), Mock()], reparameterized=False)
        output = layer.get_output_for([mean_, logvar])
        mean_grads, logvar_grads = tf.gradients(output, [mean_, logvar])
        assert(mean_grads is None)
        assert(logvar_grads is None)

        layer2 = Normal([Mock(), Mock()], reparameterized=True)
        output = layer2.get_output_for([mean_, logvar])
        mean_grads, logvar_grads = tf.gradients(output, [mean_, logvar])
        assert(mean_grads is not None)
        assert(logvar_grads is not None)

    def test_get_logpdf_for(self):
        mean_ = tf.placeholder(tf.float32, shape=(None, 1, 20))
        logvar = tf.placeholder(tf.float32, shape=(None, 1, 20))
        x = tf.placeholder(tf.float32, shape=(None, 10, 20))
        layer = Normal([Mock(), Mock()], n_samples=10)
        logpdf = layer.get_logpdf_for(x, [mean_, logvar])
        assert(logpdf.get_shape().as_list() == [None, 10])
        with tf.Session() as sess:
            test_values = sess.run(
                logpdf, feed_dict={mean_: np.random.random((2, 1, 20)),
                                   logvar: np.random.random((2, 1, 20)),
                                   x: np.random.random((2, 10, 20)) * 2})
            # TODO: test values
            assert(test_values.shape == (2, 10))

        mean_ = tf.placeholder(tf.float32, shape=(None, 1, 20, 15))
        logvar = tf.placeholder(tf.float32, shape=(None, 1, 20, 15))
        x = tf.placeholder(tf.float32, shape=(None, 10, 20, 15))
        layer = Normal([Mock(), Mock()], n_samples=10)
        logpdf = layer.get_logpdf_for(x, [mean_, logvar])
        assert(logpdf.get_shape().as_list() == [None, 10])
        with tf.Session() as sess:
            test_values = sess.run(
                logpdf, feed_dict={mean_: np.random.random((2, 1, 20, 15)),
                                   logvar: np.random.random((2, 1, 20, 15)),
                                   x: np.random.random((2, 10, 20, 15)) * 2})
            # TODO: test values
            assert(test_values.shape == (2, 10))


class TestDiscrete:
    def test_init(self):
        Discrete(Mock(), 2, n_samples=5)

    def test_get_output_for(self):
        p = tf.placeholder(tf.float32, shape=(None, 1, 20))
        layer = Discrete(Mock(), 20, n_samples=10)
        output = layer.get_output_for(p)
        assert(output.get_shape().as_list() == [None, 10, 20])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={p: np.random.random((2, 1, 20))})
            assert(test_values.shape == (2, 10, 20))
        output_d = layer.get_output_for(p, deterministic=True)
        assert(output_d == p)

        p = tf.placeholder(tf.float32, shape=(None, 5, 20))
        layer = Discrete(Mock(), 20)
        output = layer.get_output_for(p)
        assert(output.get_shape().as_list() == [None, 5, 20])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={p: np.random.random((2, 5, 20))})
            assert(test_values.shape == (2, 5, 20))

        p = tf.placeholder(tf.float32, shape=(None, 5, 20))
        layer = Discrete(Mock(), 10)
        with pytest.raises(ValueError):
            _ = layer.get_output_for(p)

    def test_get_output_for_n_samples_dynamic(self):
        p = tf.placeholder(tf.float32, shape=(None, 1, 20))
        n_samples = tf.placeholder(tf.float32, shape=())
        layer = Discrete(Mock(), 20, n_samples)
        output = layer.get_output_for(p)
        assert(output.get_shape().as_list() == [None, None, 20])
        with tf.Session() as sess:
            test_values = sess.run(
                output, feed_dict={p: np.random.random((2, 1, 20)),
                                   n_samples: 10})
            assert(test_values.shape == (2, 10, 20))

    def test_get_logpdf_for(self):
        p = tf.placeholder(tf.float32, shape=(None, 1, 10))
        x = tf.placeholder(tf.float32, shape=(None, 5, 10))
        layer = Discrete(Mock(), 10, n_samples=5)
        logpdf = layer.get_logpdf_for(x, p)
        assert(logpdf.get_shape().as_list() == [None, 5])
        x_values = np.zeros((10, 10))
        x_values[np.arange(10), np.arange(10)] = 1
        x_values = x_values.reshape((2, 5, 10))
        with tf.Session() as sess:
            test_values = sess.run(
                logpdf, feed_dict={p: np.tile(np.random.random((1, 1, 10)),
                                              (2, 1, 1)),
                                   x: x_values})
            assert(test_values.shape == (2, 5))
            assert(np.abs(np.sum(np.exp(test_values)) - 1.) < 1e-6)

    def test_get_logpdf_for_n_classes_mismatch(self):
        p = tf.placeholder(tf.float32, shape=(None, 1, 20))
        x = tf.placeholder(tf.float32, shape=(None, 5, 20))
        layer = Discrete(Mock(), 10)
        with pytest.raises(ValueError):
            _ = layer.get_logpdf_for(x, p)


class TestPrettyTensor:
    def test_init(self):
        PrettyTensor({'input': Mock()},
                     pt.template('input').fully_connected(500))
        with pytest.raises(TypeError):
            PrettyTensor({'input': Mock()}, pt.wrap(tf.placeholder(
                tf.float32, name='input')).fully_connected)

    def test_get_output_for(self):
        layer = PrettyTensor({'input': Mock()},
                             pt.template('x').fully_connected(500))
        with pytest.raises(ValueError):
            layer.get_output_for([tf.placeholder(tf.float32, (None, 100))])

        layer = PrettyTensor({'input': Mock()},
                             pt.template('input').fully_connected(500))
        assert(isinstance(
            layer.get_output_for([tf.placeholder(
                tf.float32, shape=(None, 100))]), tf.Tensor))


class TestListLayer:
    def test_init(self):
        _ = ListLayer([Mock(), Mock()])

    def test_get_output_for(self):
        a, b = Mock(), Mock()
        layer = ListLayer([Mock()])
        assert([a, b] == layer.get_output_for([a, b]))


class TestListIndexLayer:
    def test_init(self):
        index = Mock()
        layer = ListIndexLayer(Mock(), index)
        assert(layer.index == index)

    def test_get_output_for(self):
        layer = ListIndexLayer([Mock()], 1)
        a, b = Mock(), Mock()
        assert(layer.get_output_for([a, b]) == b)


def test_get_all_layers():
    in1 = InputLayer((None, 500), name='in1')
    l1 = Layer(in1, name='l1')
    in2 = InputLayer((None, 500), name='in2')
    l2 = Layer(in2, name='l2')
    l3 = MergeLayer([l1, l2], name='l3')
    l4 = MergeLayer([l3, l2], name='l4')
    layers = get_all_layers(l4)
    assert(list(map(lambda x: x.name, layers)) ==
           ['in1', 'l1', 'in2', 'l2', 'l3', 'l4'])
    layers = get_all_layers([l2, l4])
    assert(list(map(lambda x: x.name, layers)) ==
           ['in2', 'l2', 'in1', 'l1', 'l3', 'l4'])

    layers = get_all_layers(l4, treat_as_inputs=[l1])
    assert(list(map(lambda x: x.name, layers)) ==
           ['l1', 'in2', 'l2', 'l3', 'l4'])
    layers = get_all_layers(l4, treat_as_inputs=[l2, l3])
    assert(list(map(lambda x: x.name, layers)) ==
           ['l3', 'l2', 'l4'])
    layers = get_all_layers([l2, l4], treat_as_inputs=[l1])
    assert(list(map(lambda x: x.name, layers)) ==
           ['in2', 'l2', 'l1', 'l3', 'l4'])


@pytest.fixture
def _build_test_net(n_samples):
    associated_input = tf.placeholder(tf.float32, (None, 4),
                                      name='associated_input')
    l_in = InputLayer((None, 4), associated_input, name='in1')
    l1 = PrettyTensor({'l_in': l_in}, pt.template('l_in').
                      fully_connected(5),
                      name='l1')
    l1_mean = PrettyTensor({'l1': l1},
                           pt.template('l1').fully_connected(5).
                           reshape((-1, 1, 5)),
                           name='l1_mean')
    l1_logvar = PrettyTensor({'l1': l1},
                             pt.template('l1').fully_connected(5).
                             reshape((-1, 1, 5)),
                             name='l1_logvar')
    l2 = Normal([l1_mean, l1_logvar], n_samples, name='l2')
    l3 = PrettyTensor({'l2': l2},
                      pt.template('l2').reshape((-1, 5)).
                      fully_connected(10).
                      reshape((-1, n_samples, 10)),
                      name='l3')
    l4 = PrettyTensor({'l_in': l_in},
                      pt.template('l_in').fully_connected(10).
                      reshape((-1, 1, 10)),
                      name='l4')
    l5 = Normal([l3, l4], name='l5')
    output_layers = [l1, l1_mean, l1_logvar, l2, l3, l4, l5]

    return output_layers


@pytest.fixture
def _assert_outputs(outputs, n_samples):
    o1, o1_mean, o1_logvar, o2, o3, o4, o5 = list(map(lambda x: x[0], outputs))
    pd1, pd1_mean, pd1_logvar, pd2, pd3, pd4, pd5 = \
        list(map(lambda x: x[1], outputs))
    assert (o1.get_shape().as_list() == [None, 5])
    assert (o1_mean.get_shape().as_list() == [None, 1, 5])
    assert (o1_logvar.get_shape().as_list() == [None, 1, 5])
    assert (o2.get_shape().as_list() == [None, n_samples, 5])
    assert (o3.get_shape().as_list() == [None, n_samples, 10])
    assert (o4.get_shape().as_list() == [None, 1, 10])
    assert (o5.get_shape().as_list() == [None, n_samples, 10])
    assert ([pd1, pd1_mean, pd1_logvar, pd3, pd4] == [None] * 5)
    assert (pd2.get_shape().as_list() == [None, n_samples])
    assert (pd5.get_shape().as_list() == [None, n_samples])


def test_get_output_n_samples_dynamic():
    n_samples = tf.placeholder(tf.int32, shape=())
    output_layers = _build_test_net(n_samples)
    outputs = get_output(output_layers)
    _assert_outputs(outputs, None)


def test_get_output_n_samples_int():
    n_samples = 3
    output_layers = _build_test_net(n_samples)
    outputs = get_output(output_layers)
    _assert_outputs(outputs, 3)


def test_get_output_feed_input_layer():
    n_samples = 3
    output_layers = _build_test_net(n_samples)
    l_in_feed = tf.placeholder(tf.float32, shape=(None, 4))
    outputs = get_output(output_layers, l_in_feed)
    _assert_outputs(outputs, n_samples)


def test_get_output_feed_2_layers():
    n_samples = 3
    output_layers = _build_test_net(3)
    l1, l1_mean, l1_logvar, l2, l3, l4, l5 = output_layers
    l1_feed = tf.placeholder(tf.float32, shape=(None, 5))
    l2_feed = tf.placeholder(tf.float32, shape=(None, n_samples, 5))
    inputs = {l1: l1_feed, l2: l2_feed}
    outputs = get_output(output_layers, inputs)
    _assert_outputs(outputs, n_samples)
    assert(outputs[0][0] == l1_feed)
    assert(outputs[3][0] == l2_feed)
    assert(outputs[3][1].get_shape().as_list() == [None, n_samples])
    assert(outputs[6][1].get_shape().as_list() == [None, n_samples])


def test_get_output_single_output():
    n_samples = tf.placeholder(tf.int32, shape=())
    output_layers = _build_test_net(n_samples)
    l1, l1_mean, l1_logvar, l2, l3, l4, l5 = output_layers
    l2_feed = tf.placeholder(tf.float32, shape=(None, None, 5))
    l5_feed = tf.placeholder(tf.float32, shape=(None, None, 10))
    inputs = {l2: l2_feed, l5: l5_feed}
    output = get_output(l5, inputs)
    assert(output[0] == l5_feed)
    assert(output[1].get_shape().as_list() == [None, None])


def test_get_output_raise():
    mean = InputLayer((None, 1, 5))
    logvar = InputLayer((None, 1, 5))
    layer = Normal([mean, logvar], 3)
    mean_feed = tf.placeholder(tf.float32, shape=(None, 1, 5))
    with pytest.raises(ValueError) as e:
        get_output(layer, {mean: mean_feed})
    assert(e.value.message.find("get_output() was called without giving an "
                                "input expression for the InputLayer") >= 0)

    with pytest.raises(ValueError) as e:
        get_output(layer, mean_feed)
    assert(e.value.message == "get_output() was called with a single input "
                              "expression on a network with multiple input "
                              "layers. Please call it with a dictionary of "
                              "input expressions instead.")

    layer = Normal([mean, None], 3)
    with pytest.raises(ValueError) as e:
        get_output(layer, mean_feed)
    assert(e.value.message.find("get_output() was called with free-floating "
                                "layer") >= 0)


def test_get_output_discrete_layer():
    p = InputLayer((None, 1, 5))
    layer = Discrete(p, 5, n_samples=3, name='discrete')
    p_feed = tf.placeholder(tf.float32, shape=(None, 1, 5))
    output = get_output(layer, p_feed)
    assert(output[0].get_shape().as_list() == [None, 3, 5])


def test_get_output_deterministic():
    mean = InputLayer((None, 1, 5))
    logvar = InputLayer((None, 1, 5),
                        tf.placeholder(tf.float32, (None, 1, 5)))
    layer = Normal([mean, logvar])
    mean_feed = tf.placeholder(tf.float32, shape=(None, 1, 5))
    output = get_output(layer, {mean: mean_feed}, deterministic=True)
    assert(output == mean_feed)


def test_get_output_nonexist_args():
    mean = InputLayer((None, 1, 5))
    logvar = InputLayer((None, 1, 5),
                        tf.placeholder(tf.float32, (None, 1, 5)))
    layer = Normal([mean, logvar])
    mean_feed = tf.placeholder(tf.float32, shape=(None, 1, 5))
    with pytest.warns(RuntimeWarning) as e:
        get_output(layer, {mean: mean_feed}, determinstic=True)
    assert(e.pop().message.message.find('deterministic') >= 0)
    with pytest.warns(RuntimeWarning):
        get_output(layer, {mean: mean_feed}, non_exist_arg=None)
