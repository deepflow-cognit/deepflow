from setuptools import setup

setup(
   name='deepflow',
   version='1.6',
   description='deepflow is a standard neural networking package made for creating layers and training data.',
   license="Apache",
   author='Aaha3',
   maintainer='Aaha3',
   url='https://deepflow-cognit.github.io/deepflow-cognit.org/',
   packages=['deepflow'],  #same as name
   install_requires=['numpy','uuid'], #external packages as dependencies
)