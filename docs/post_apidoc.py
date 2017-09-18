#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re


api_dir = './api'

# add automodule options for distributions and StochasticTensor
options = [':inherited-members:']
modules = ['zhusuan.distributions.univariate',
           'zhusuan.distributions.multivariate',
           'zhusuan.model.stochastic',
           'zhusuan.variational.exclusive_kl',
           'zhusuan.variational.monte_carlo',
           'zhusuan.variational.inclusive_kl']

for module in modules:
    module_path = os.path.join(api_dir, module + '.rst')
    with open(module_path, 'r') as f:
        module_string = f.read()
    target = r'\:members\:(\n|.)*\:undoc-members\:(\n|.)*\:show-inheritance\:'
    indent = '    '
    rep = ':members:\n' + indent + ':undoc-members:\n' + indent + \
        ':show-inheritance:'
    for option in options:
        rep += '\n' + indent + option
    post_module_string = re.sub(target, rep, module_string)
    with open(module_path, 'w') as f:
        f.write(post_module_string)
