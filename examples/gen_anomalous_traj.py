import trajpy.traj_generator as tjg
import numpy as np
"""
    generate N anomalous trajectories for each value of alpha exponents
    ranging between 0.10 and 2.10
"""

n_steps = 250  # number of time steps
n_samples = 1  # number of trajectories
dt = 1.0  # time increment
alphas = np.linspace(0.10, 2.1, 20)

for value in alphas:

    xa, ya = tjg.anomalous_diffusion(n_steps, n_samples, dt, alpha=value)

    tjg.save_to_file(ya, value, 'data/anomalous')
