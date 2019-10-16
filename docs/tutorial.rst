Tutorial
==================================


Simple Example
--------
::
    import trajpy.trajpy as tj
    import numpy as np
    import matplotlib.pyplot as plt

    filename = '/data/samples/sample.csv'
    r = tj.Trajectory(filename)
    r.compute_all()
    plt.plot(r._t, r.msd_ea)
    plt.show()

Features
--------



Training data
--------------