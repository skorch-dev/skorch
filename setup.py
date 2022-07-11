import os

from setuptools import setup, find_packages


with open('VERSION', 'r') as f:
    version = f.read().rstrip()

with open('requirements.txt') as f:
    install_requires = [l.strip() for l in f]


python_requires = '>=3.7'

tests_require = [
    'pytest',
    'pytest-cov',
]

docs_require = [
    'Sphinx',
    'sphinx_rtd_theme',
    'numpydoc',
]

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
except IOError:
    README = ''


setup(
    name='skorch',
    version=version,
    description='scikit-learn compatible neural network library for pytorch',
    long_description=README,
    license='new BSD 3-Clause',
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/skorch-dev/skorch",
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        'docs': docs_require,
    },
)
