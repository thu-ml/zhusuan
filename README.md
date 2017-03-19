# ZhuSuan

A Library for Generative Models.

## Supported Inference
### (Stochastic) Variational Inference (VI & SVI)
* Kinds of variational posteriors we support:
  * __Mean-field__ posterior: Fully-factorized.
  * __Structured__ posterior: With user specified dependencies.

* Variational objectives we support:
  * __ADVI__: Automatic Differentiation Variational Inference
  * __IWAE__: Importance weighted objectives
  * __NVIL__: Score function estimator with variance reduction
  * __VIMCO__: Multi-sample score function estimator with variance reduction

### Adaptive Importance Sampling
* Reweighted Wake-sleep (__RWS__): With user specified adaptive proposal.

### Markov Chain Monte Carlo (MCMC)
* Hamiltonian Monte Carlo (__HMC__): With step size and mass adaptation.

## Examples
* Gaussian: 
[HMC](examples/gaussian.py)
* Toy 2D Intractable Posterior: 
[ADVI](examples/toy2d.py)
* Beyesian Neural Networks: 
[ADVI](examples/bayesian_nn.py)
* Variational Autoencoder (VAE): 
[ADVI](examples/vae.py), 
[IWAE](examples/iwae.py)
* Convolutional VAE: 
[ADVI](examples/vae_conv.py)
* Semi-supervised VAE (Kingma, 2014): 
[ADVI](examples/vae_ssl.py),
[RWS](examples/vae_ssl_rws.py)
* Deep Sigmoid Belief Networks
[RWS](examples/sbn_rws.py),
[VIMCO](examples/sbn_vimco.py)
* Logistic Normal Topic Model: 
[HMC](examples/lntm_mcem.py)

## Developments

### Tests

First install requirements for developments.

`pip install -r requirements-dev.txt`

This command will run automatic tests in the main directory.

`python -m unittest discover -v`

##### Test Coverage
After running tests, to ensure test coverage keeping 100% over the developments, run

`coverage report --include="zhusuan/*"`

##### PEP8 Code Style Check
We follow PEP8 python code style. To check, in the main directory, run

`pep8 .`
