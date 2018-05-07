#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.MD') as readme_file:
    readme = readme_file.read()

requirements = [
    'torch', 'matplotlib', 'numpy',
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

setup(
    name='deep_ensemble',
    version='0.1.0',
    description="Deep Ensemble implementation",
    long_description=readme,
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
