# ZhuSuan

A Framework of Probabilistic Inference, Bayesian Modeling and Deep Generative Models.

## Supported Inference
### (Stochastic) Variational Inference (VI & SVI)
* Kinds of variational posteriors we support:
  * __Mean-field__ posterior: Fully-factorized.
  * __Structured__ posterior: With user specified dependencies.

* Variational objectives we support:
  * Automatic Differentiation Variational Inference (__ADVI__)
  * Importance weighted objectives (__IWAE__)

### Particle Belief Propagation
* Reweighted Wake-sleep (__RWS__): With user specified adaptive proposal.

### Markov Chain Monte Carlo (MCMC)
* Hamiltonian Monte Carlo (__HMC__)
* No-U-Turn Sampler (__NUTS__)

## Examples
* Toy intractable posterior: [ADVI](https://github.com/thu-ml/ZhuSuan/blob/master/examples/toy.py)
* Bayesian logistic regression: [NUTS](https://github.com/thu-ml/ZhuSuan/blob/master/examples/blr.py)
* Bayesian Multinomial logistic regression: [NUTS](https://github.com/thu-ml/ZhuSuan/blob/master/examples/bmlr.py)
* Beyesian neural networks: [ADVI](https://github.com/thu-ml/ZhuSuan/blob/master/examples/bayesian_nn.py), NUTS

### Deep Generative Models
* Variational autoencoder (VAE): [ADVI](https://github.com/thu-ml/ZhuSuan/blob/master/examples/vae.py)
* Convolutional VAE: [ADVI](https://github.com/thu-ml/ZhuSuan/blob/master/examples/vae_conv.py)
* Semi-supervised VAE (Kingma, 2014): 
  [ADVI](https://github.com/thu-ml/ZhuSuan/blob/master/examples/vae_ssl.py),
  [RWS](https://github.com/thu-ml/ZhuSuan/blob/master/examples/vae_ssl_rws.py)
* DRAW (Gregor, 2015): [ADVI](https://github.com/thu-ml/ZhuSuan/blob/master/examples/draw.py)

## Developments

### Tests
This command will run automatic tests in the tests/ directory.

`py.test`

##### Test Coverage
To ensure test coverage keeping 100% over the developments, run

`py.test --cov zhusuan/ --cov-report term-missing`

##### PEP8 Code Style Check
We follow PEP8 python code style. To check, run

`py.test --pep8`
