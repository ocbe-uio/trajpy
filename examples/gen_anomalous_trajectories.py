import trajpy.traj_generator as tjg
import numpy as np
"""
    generate N anomalous trajectories for each value of alpha exponents
    ranging between 0.10 and 2.10
"""

t_steps = 250
n_sample = 1
dt = 1.
alphas = np.linspace(0.10, 2.1, 20)

for value in alphas:

    xa, ya = tjg.anomalous_diffusion(t_steps, n_sample, dt, alpha=value)

    for i in range(0, n_sample):
        np.savetxt('data/trj' + str(np.round(value, decimals=1)) + str(i) + '.csv', ya[:, i], delimiter=",", header='m')
        print(np.round(value, decimals=2), i)
