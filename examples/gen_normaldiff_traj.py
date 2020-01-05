import trajpy.traj_generator as tjg
import numpy as np
"""
    generate N normal diffusion trajectories for each value of diffusion coefficient
"""

n_steps = 250  # number of time steps
n_samples = 1  # number of trajectories
dt = 1.0  # time increment
diffusivity = np.array([10., 100., 1000., 10000., ])

for value in diffusivity:

    xa, ya = tjg.normal_diffusion(n_steps, n_samples, 1.0, 0., value, dt)

    for n in range(0, n_samples):
        np.savetxt('data/trj' + str(np.round(value, decimals=1)) + str(n) + '.csv', ya[:, n], delimiter=",", header='m')
        print(np.round(value, decimals=2), n)