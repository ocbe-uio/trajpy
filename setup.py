"""
this setup will check for dependencies and install TrajPy on your computer
"""
from setuptools import setup, find_packages

setup(
    name='trajpy',
    version='1.4.0',
    url='https://github.com/ocbe-uio/trajpy.git',
    author='Mauricio Moreira and Eduardo Mossmann',
    author_email='trajpy@protonmail.com',
    description='Trajectory classifier for cells, nanoparticles & whatelse.',
    keywords=['trajectory quantification', 'feature engineering', 'diffusion classification'],
    license='GNU GPLv3',
    platform='Python 3.7',
    packages=find_packages(),
    install_requires=['numpy >= 1.14.3',
                      'scipy == 1.7.1'],
)
