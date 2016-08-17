# ZhuSuan

A Framework of Probabilistic Inference, Bayesian Modeling and Deep Generative Models.

## Examples

### Examples of ADVI (Automatic Differentiation Variational Inference)
* Toy intractable posterior: `python examples/toy.py`
* Variational autoencoder: `python examples/vae.py`


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
