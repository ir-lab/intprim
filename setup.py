from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import sys

USE_CYTHON = True
COMPILE_CYTHON = False

if USE_CYTHON:
    ext = '.pyx' if COMPILE_CYTHON else '.c'

    extensions = [Extension('basis_model', ['intprim/basis_model' + ext]),
                  Extension('bayesian_interaction_primitives', ['intprim/bayesian_interaction_primitives' + ext])]

    if COMPILE_CYTHON:
        from Cython.Build import cythonize
        extensions = cythonize(extensions)

    setup(
        name = 'intprim',
        version = '1.0',
        description = 'Interaction Primitives library from the Interactive Robotics Lab at Arizona State University',
        url = 'https://github.com/ir-lab/interaction-primitives',
        author = 'Joseph Campbell',
        author_email = 'jacampb1@asu.edu',
        license = 'MIT',
        packages = ['intprim', 'intprim.examples'],
        ext_modules = extensions,
        include_dirs=[np.get_include()]
    )
else:
    setup(
        name='intprim',
        version='1.0',
        description='Interaction Primitives library from the Interactive Robotics Lab at Arizona State University',
        url='https://github.com/ir-lab/interaction-primitives',
        author='Joseph Campbell',
        author_email='jacampb1@asu.edu',
        license='MIT',
        packages=['intprim', 'intprim.examples']
    )
