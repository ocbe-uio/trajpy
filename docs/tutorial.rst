Tutorial
==================================


Simple Example
--------
::
  import trajpy.trajpy as tj
  import numpy as np
  import matplotlib.pyplot as plt

  filename = 'data/samples/sample.csv'
  r = tj.Trajectory(filename,
                    skip_header=1,
                    delimiter=',')

  r.compute_features()


Accessing Features
--------

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

Training data
--------------