[![PyPI version](https://badge.fury.io/py/trajpy.svg)](https://badge.fury.io/py/trajpy)
[![Maintainability](https://api.codeclimate.com/v1/badges/650cde37de8ccb468b8c/maintainability)](https://codeclimate.com/github/phydev/trajpy/maintainability)
[![codecov](https://codecov.io/gh/ocbe-uio/trajpy/branch/master/graph/badge.svg?token=lhYwQjiAlU)](https://codecov.io/gh/ocbe-uio/trajpy)
[![Build Status](https://travis-ci.com/ocbe-uio/trajpy.svg?branch=master)](https://travis-ci.com/ocbe-uio/trajpy)
[![PyUp](https://pyup.io/repos/github/ocbe-uio/trajpy/shield.svg?t=1570846676802)](https://pyup.io/repos/github/ocbe-uio/trajpy/)
[![Python 3](https://pyup.io/repos/github/ocbe-uio/trajpy/python-3-shield.svg)](https://pyup.io/repos/github/ocbe-uio/trajpy/)
[![Documentation Status](https://readthedocs.org/projects/trajpy/badge/?version=latest)](https://trajpy.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/194252287.svg)](https://zenodo.org/badge/latestdoi/194252287)


# TrajPy

Trajectory classification is a challenging task and fundamental for
analysing the movement of nanoparticles, bacteria, cells and active
matter in general.

We propose TrajPy as an easy pythonic solution to be applied in studies that
demand trajectory classification. It requires little knowledge of programming
and physics to be used by nonspecialists.

TrajPy is composed of three main units of code:

- The training data set is built using a **trajectory generator**
- **Features** are computed for characterizing the trajectories
- The **classifier** built on Scikit-Learn.

Our dataset and Machine Learning (ML) model are available for use, as well
the generator for building your own database.

## Installation


We have the package hosted at PyPi, for installing use the command line: 
```bash
pip3 install trajpy
```
If you want to test the development version, clone the repository at your local directory from your terminal:
```bash
git clone https://github.com/phydev/trajpy
```
Then run the setup.py for installing 
```bash
python setup.py --install
```

## Basic Usage Example
First we import the package 
```python
import trajpy.trajpy as tj
```
Then we load the data sample provided in this repository, we pass the arguments `skip_header=1`
 to skip the first line of the file and `delimiter=','` to specify the file format
``` python
filename = 'data/samples/sample.csv'
r = tj.Trajectory(filename,
                  skip_header=1,
                  delimiter=',')
```
Finally, for computing a set of features for trajectory analysis we can simple run the function `r.compute_features()`
```python
    r.compute_features()
```
The features will be stored in the object `r`, for instance:
```python
  >>> r.asymmetry
  >>> 0.5782095322093505
  >>> r.fractal_dimension
  >>> 1.04
  >>> r.efficiency
  >>> 0.29363293632936327
  >>> r.gyration_radius
  >>> array([[30.40512689,  5.82735002,  0.96782673],
  >>>     [ 5.82735002,  2.18625318,  0.27296851],
  >>>     [ 0.96782673,  0.27296851,  2.41663589]])
```

For more examples please consult the extended documentation: https://trajpy.readthedocs.io/

## Requirements

- numpy >= 1.14.3
- scipy >= 1.5.4

[ ~ Dependencies scanned by PyUp.io ~ ]

## Citation
If using the TrajPy package in academic work, please cite Moreira-Soares et al. (2020), in addition to the relevant methodological papers.

```latex
@article{moreira2020adhesion,
  title={Adhesion modulates cell morphology and migration within dense fibrous networks},
  author={Moreira-Soares, Maur{\'\i}cio and Cunha, Susana P and Bordin, Jos{\'e} Rafael and Travasso, Rui DM},
  journal={Journal of Physics: Condensed Matter},
  volume={32},
  number={31},
  pages={314001},
  year={2020},
  publisher={IOP Publishing}
}

@software{mauricio_moreira_2020_3978699,
  author       = {Mauricio Moreira and Eduardo Mossmann},
  title        = {phydev/trajpy: TrajPy 1.3.1},
  month        = aug,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.3.1},
  doi          = {10.5281/zenodo.3978699},
  url          = {https://doi.org/10.5281/zenodo.3978699}
}
```

## Contribution
This is an open source project, and all contributions are welcome. Feel free to open an Issue, a Pull Request, or to e-mail us.

## Publications using trajpy
 Simões, RF, Pino, R, Moreira-Soares, M, et al. [Quantitative Analysis of Neuronal Mitochondrial Movement Reveals Patterns Resulting from Neurotoxicity of Rotenone and 6-Hydroxydopamine.](https://faseb.onlinelibrary.wiley.com/doi/10.1096/fj.202100899R) FASEB J. 2021; 35:e22024. doi:10.1096/fj.202100899R

Moreira-Soares, M., Pinto-Cunha, S.,  Bordin, J. R., Travasso, R. D. M. *[Adhesion modulates cell morphology and migration within dense fibrous networks](https://www.biorxiv.org/content/10.1101/838995v1)*.  https://doi.org/10.1088/1361-648X/ab7c17

## References
Arkin, H. and Janke, W. 2013. Gyration tensor based analysis of the shapes of polymer chains in an attractive spherical cage. J Chem Phys 138, 054904.

Wagner, T., Kroll, A., Haramagatti, C.R., Lipinski, H.G. and Wiemann, M. 2017. Classification and Segmentation of Nanoparticle Diffusion Trajectories in Cellular Micro Environments. PLoS One 12, e0170165.
