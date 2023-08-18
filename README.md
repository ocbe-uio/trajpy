[![PyPI version](https://badge.fury.io/py/trajpy.svg)](https://badge.fury.io/py/trajpy)
![Build Status](https://github.com/ocbe-uio/trajpy/actions/workflows/python-app.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/trajpy/badge/?version=latest)](https://trajpy.readthedocs.io/en/latest/?badge=latest)
[![Python3](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/) 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/194252287.svg)](https://zenodo.org/badge/latestdoi/194252287)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ocbe-uio/trajpy/master?labpath=%2Fexamples%2Fclinical_data_analysis_tutorial.ipynb)

# TrajPy

Trajectory analysis is a challenging task and fundamental for
understanding the movement of living organisms in various scales.

We propose TrajPy as an easy pythonic solution to be applied in studies that
demand trajectory analysis. With a friendly graphic user interface (GUI) it requires little knowledge of computing
and physics to be used by nonspecialists.

TrajPy is composed of three main units of code:

- Basic usage:
  - The GUI: it is where you interact with trajpy and the only thing you need to know to start using it
- Advanced  
  - trajpy.py: it's the heart of trajpy, it computes the **Features** for characterizing the trajectories 
  - traj_generator.py: a **trajectory generator** that can be used to build a dataset for trajectory classification


Our dataset and Machine Learning (ML) model are available for use, as well
the generator for building your own database.

## Installation


We have the package hosted at PyPi, for installing use the command line: 
```bash
pip3 install trajpy
```
If you want to test the development version, clone the repository at your local directory from your terminal:
```bash
git clone https://github.com/ocbe-uio/trajpy
```
Then run the setup.py for installing 
```bash
python setup.py --install
```

## Basic Usage Example

### Using the Graphic User Interface (GUI)

Open a terminal and execute the line bellow
```bash
python3 -m trajpy.gui
```

1 - You can open one file at time clicking on `Open file...` or process several files in the same director with `Open directory...`

2 - Select the features to be computed by ticking the boxes

3 - Click on `Compute`

4 - Select the directory and file name where the results will be stored

The processing is ready when the following message appears in the text box located at the bottom of the GUI:

`Results saved to /path/to/results/output.csv`

### File formats

#### Comma separated values (CSV)
Currently trajpy support CSV files organized in 4 columns: time `t` and 3 spatial coordinates `x`, `y`, `z`:

|t|x|y|z|
|---|---|---|---|
| 1.00 | 10.00 | 50.00 | 50.00
| 2.00 | 11.00 | 50.00 | 50.00
| 3.00 | 11.00 | 50.00 | 50.00
| 4.00 | 12.00 | 50.00 | 50.00
| 5.00 | 12.00 | 50.00 | 50.00
| 6.00 | 13.00 | 50.00 | 50.00

See the [sample file](https://github.com/ocbe-uio/trajpy/blob/a370e49444ea845becb573fd5cc835b5c899c7dc/data/samples/sample.csv) provided in this repository as example.

#### LAMMPS YAML dump format

LAMMPS YAML files are defined with the following structure:
```yaml
    ---
    time: 0.0
    natoms: 100
    keywords: [id, type, x, y, z, vx, vy, vz, fx, fy, fz]
    data:
    - [1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -nan, -nan, -nan]
    - [2, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -nan, -nan, -nan]
    - [3, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -nan, -nan, -nan]
    ...
```
We provide support for parsing this type of data files with the function [`parse_lammps_dump_yaml()`](https://github.com/ocbe-uio/trajpy/blob/8381bedfc3f0d696072af1d66f08af497eb0cced/trajpy/auxiliar_functions.py#L5).


### Scripting

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
- scipy >= 1.7.1
- ttkthemes >= 2.4.0
- Pillow >= 8.1.0
- PyYAML >= 5.3.1


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

Eduardo Henrique Mossmann. [A physics based feature engineering framework for trajectory analysis](#). MSc dissertation. Federal University of Pelotas 2022,  Brazil.

 Simões, RF, Pino, R, Moreira-Soares, M, et al. [Quantitative Analysis of Neuronal Mitochondrial Movement Reveals Patterns Resulting from Neurotoxicity of Rotenone and 6-Hydroxydopamine.](https://faseb.onlinelibrary.wiley.com/doi/10.1096/fj.202100899R) FASEB J. 2021; 35:e22024. doi:10.1096/fj.202100899R

Moreira-Soares, M., Pinto-Cunha, S.,  Bordin, J. R., Travasso, R. D. M. *[Adhesion modulates cell morphology and migration within dense fibrous networks](https://www.biorxiv.org/content/10.1101/838995v1)*.  https://doi.org/10.1088/1361-648X/ab7c17

## References
Arkin, H. and Janke, W. 2013. Gyration tensor based analysis of the shapes of polymer chains in an attractive spherical cage. J Chem Phys 138, 054904.

Wagner, T., Kroll, A., Haramagatti, C.R., Lipinski, H.G. and Wiemann, M. 2017. Classification and Segmentation of Nanoparticle Diffusion Trajectories in Cellular Micro Environments. PLoS One 12, e0170165.
