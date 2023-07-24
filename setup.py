from setuptools import setup, find_packages

setup(
    name='rl',
    packages = ['models', 'commons'],
    version='0.0.2',
    url='https://github.com/5121eun/rl.git',
    install_requires=[
        'torch'
    ]
)
