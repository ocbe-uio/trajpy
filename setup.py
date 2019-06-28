'''
this setup will check for dependencies and install TrajPy on your computer
'''
from setuptools import setup, find_packages

setup(
    name = 'TrajPy',
    version = '0.0.1',
    url = 'https://gitlab.com/phydev/trajpy.git',
    author = 'Mauricio Moreira',
    author_email = 'mms@uc.pt',
    description = 'Trajectory classifier for cell research',
    license = 'GNU GPLv3',
    platform = 'Python 3.6',
    packages = find_packages(),
    install_requires = ['numpy >= 1.14.3',
                        'matplotlib >= 3.0.3',
                        'scipy >= 1.3.0',
                        'scikit-learn >= 0.20.3'],
)
