.. ZhuSuan documentation master file, created by
   sphinx-quickstart on Wed Feb  8 15:01:57 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ZhuSuan
=======

ZhuSuan is a python	library	for	**Generative Models**, built upon
`Tensorflow <https://www.tensorflow.org>`_.
Unlike existing deep learning libraries, which are mainly designed for
supervised tasks, ZhuSuan is featured for its deep root into Bayesian
Inference, thus supporting various kinds of generative models: both the
traditional **hierarchical Bayesian models** and recent
**deep generative models**.

With ZhuSuan, users can enjoy powerful fitting and multi-GPU training of deep
learning, while at the same time they can use generative models to model the
complex world, exploit unlabeled data and deal with uncertainty by performing
principled Bayesian inference.

.. toctree::
   :maxdepth: 2


Installation
------------

ZhuSuan is still under development. Before the first stable release (1.0),
please clone the `GitHub repository <https://github.com/thu-ml/zhusuan>`_ and
run
::

   pip install .

in the main directory. This will install ZhuSuan and its dependencies
automatically. ZhuSuan also requires Tensorflow version 1.0 or later. Because
users should choose whether to install the cpu or gpu version of Tensorflow,
we do not include it in the dependencies. See
`Installing Tensorflow <https://www.tensorflow.org/install/>`_.

If you are developing ZhuSuan, you may want to install in an "editable" or
"develop" mode. Please refer to the Contribution section in
`README <https://github.com/thu-ml/zhusuan/blob/master/README.rst>`_.

After installation, open your python console and type::

   >>> import zhusuan as zs

If no error occurs, you've successfully installed ZhuSuan.


Tutorials
---------

.. toctree::
   :maxdepth: 2

   tutorials


API Docs
--------

Information on specific functions, classes, and methods.

.. toctree::
   :maxdepth: 2

   api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
