# ZhuSuan

A Framework of Probabilistic Inference, Bayesian Modeling and Deep Generative Models.

## Examples

### Examples of Variational Inference (ADVI, etc.)
* Toy intractable posterior: `python examples/toy.py`
* Variational autoencoder (VAE): `python examples/vae.py`
* Convolutional VAE: `python examples/vae_conv.py`
* Semi-supervised VAE: `python examples/vae_ssl.py`

### Examples of MCMC (NUTS, etc.)
* Bayesian logistic regression: `python examples/blr.py`

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
