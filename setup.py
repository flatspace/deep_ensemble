#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: put package requirements here
]

setup_requirements = [
    'pytest-runner',
    # TODO(flatspace): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='deep_ensemble',
    version='0.1.0',
    description="Deep Ensemble implementation",
    long_description=readme + '\n\n' + history,
    author="Brecht Dierckx",
    author_email='brecht.dierckx@gmail.com',
    url='https://github.com/flatspace/deep_ensemble',
    packages=find_packages(include=['deep_ensemble']),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='deep_ensemble',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
