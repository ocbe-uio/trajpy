import numpy as np


def weierstrass_mandelbrot(t, n_displacements, gamma=np.sqrt(np.pi), alpha=.8):
    """
    calculates the weierstrass mandelbrot function
    
    .. math::
        W(t) = \\sum_{-\\infty}^{\\infty} \\frac{\\cos{(\\phi_n )} - \\cos{(\\gamma^n t^* + \\phi_n )} }{\\gamma^{n\\alpha/2}}}

    :param t: time step
    :param n_displacements: number of displacements
    :param gamma: sqrt(pi)
    :param alpha: anomalous exponent
    :return: anomalous step
    """

    t_star = (2. * np.pi * t) / n_displacements

    wsm = 0.

    for iteration in range(-8, 49):  # [-8, 48]
        phi = 2. * np.random.rand() * np.pi
        wsm += (np.cos(phi) - np.cos(np.power(gamma, iteration) * t_star + phi)) / \
               (np.power(gamma, iteration * (alpha / 2.)))
    return wsm


def anomalous_diffusion(n_steps, n_sample, time_step, alpha=.8):
    """
    :param n_steps: total number of steps
    :param n_sample: number of simulations
    :param time_step: time step
    :param alpha: anomalous exponent
    :return x, y: time, array containing N_sample trajectories with Nsteps
    """
    x = np.zeros(n_steps) * time_step
    y = np.zeros((n_steps, n_sample))

    for i_sample in range(0, n_sample):

        for i_step in range(0, n_steps):
            t = i_step * time_step
            y[i_step, i_sample] = weierstrass_mandelbrot(t, n_steps, alpha=alpha)
            x[i_step] = t

    if n_sample == 1:
        y = y.transpose()[0]

    return x, y
