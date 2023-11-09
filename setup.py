from setuptools import setup, find_packages

setup(
    name='my-financial-package',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'statsmodels', 'pandas'
    ],
)
