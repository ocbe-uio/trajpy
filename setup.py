"""
this setup will check for dependencies and install TrajPy on your computer
"""
from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='trajpy',
    version='1.4.3',
    url='https://github.com/ocbe-uio/trajpy.git',
    author='Mauricio Moreira',
    author_email='trajpy@protonmail.com',
    description='Empowering feature engineering for trajectory analysis across domains.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['trajectory quantification', 'feature engineering', 'diffusion classification'],
    license='GNU GPLv3',
    platform='Python 3.7',
    packages=find_packages(),
    install_requires=['numpy >= 1.14.3',
                      'scipy >= 1.7.1',
                     'ttkthemes>=2.4.0',
                        'Pillow>=8.1.0',
                     'PyYAML >= 5.3.1'],
)
