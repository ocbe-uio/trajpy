import trajpy.traj_generator as tjg
import numpy as np
"""
    generates N superdiffusion (direct motion) trajectories for each value of velocity
"""

n_steps = 250  # number of time steps
n_samples = 1  # number of trajectories
dt = 1.0  # time increment
radius = np.array([5., 10., 20.])
velocity = np.array([0.1, 1., 2., 5.])

for value in velocity:

    xa, ya = tjg.superdiffusion(value, n_steps, n_samples, 0., dt)

    tjg.save_to_file(ya, value, 'data/superdiff')
