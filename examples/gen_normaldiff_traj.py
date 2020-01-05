import trajpy.traj_generator as tjg
import numpy as np
"""
    generates N normal diffusion trajectories for each value of diffusion coefficient
"""

n_steps = 250  # number of time steps
n_samples = 1  # number of trajectories
dt = 1.0  # time increment
diffusivity = np.array([10., 100., 1000., 10000., ])

for value in diffusivity:

    xa, ya = tjg.normal_diffusion(n_steps, n_samples, 1.0, 0., value, dt)

    tjg.save_to_file(ya, value, 'data/normal')
