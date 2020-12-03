#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "bson>=0.5.10",
    "Click>=7.0",
    "dnspython>=2.0.0",
    "gensim>=3.8.3",
    "matplotlib>=3.3.3",
    "pandas>=1.1.4",
    "pymongo>=3.11.1",
    "scikit-learn>=0.23.2",
    "starlette>=0.14.1",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Leon Kozlowski",
    author_email="leonkozlowski@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Baseline multi-class text classifiers",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="baseline",
    name="baseline",
    packages=find_packages(include=["baseline", "baseline.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/leonkozlowski/baseline",
    version="0.1.0",
    zip_safe=False,
)
