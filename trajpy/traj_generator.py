import numpy as np


def weierstrass_mandelbrot(t, n_displacements, alpha):
    """
    Calculates the weierstrass mandelbrot function
    
    .. math::
        W(t) = \\sum_{n=-\\infty}^{\\infty} \\frac{\\cos{(\\phi_n )} - \\cos{(\\gamma^n t^* + \\phi_n )} }{\\gamma^{n\\alpha/2}} \\, .

    :param t: time step
    :param n_displacements: number of displacements
    :param alpha: anomalous exponent
    :return: anomalous step
    """
    gamma = np.sqrt(np.pi)
    t_star = (2. * np.pi * t) / n_displacements

    wsm = 0.

    for iteration in range(-8, 49):  # [-8, 48]
        phi = 2. * np.random.rand() * np.pi
        wsm += (np.cos(phi) - np.cos(np.power(gamma, iteration) * t_star + phi)) / \
               (np.power(gamma, iteration * (alpha / 2.)))
    return wsm


def anomalous_diffusion(n_steps, n_samples, time_step, alpha):
    """
    Generates an ensemble of anomalous trajectories.

    :param n_steps: total number of steps
    :param n_samples: number of simulations
    :param time_step: time step
    :param alpha: anomalous exponent
    :return x, y: time, array containing N_sample trajectories with Nsteps
    """
    x = np.zeros(n_steps) * time_step
    y = np.zeros((n_steps, n_samples))

    for i_sample in range(0, n_samples):

        for i_step in range(0, n_steps):
            t = i_step * time_step
            y[i_step, i_sample] = weierstrass_mandelbrot(t, n_steps, alpha=alpha)
            x[i_step] = t

    if n_samples == 1:
        y = y.transpose()[0]

    return x, y


def normal_distribution(u, D, dt):
    """
    This is the steplength probability density function for normal diffusion.

    :param u: absolute distance travelled by the particle durint the time interval dt
    :param D: diffusivity
    :param dt: time interval
    :return pdf: probability density function

    """
    diff = 4. * D * dt
    pdf = ((2. * u) / diff) * np.exp(-np.power(u, 2) / diff)
    return pdf


def normal_diffusion(n_steps, n_samples, dx, y0, D, dt):
    """
    Generates an ensemble of normal diffusion trajectories.

    :param n_steps: total steps
    :param n_samples: number of trajectories
    :param dx: maximum step length
    :param y0: starting position
    :param D: diffusivity
    :param dt: time step
    :return x, y: time, array containing N_samples trajectories with N_steps
    """
    
    y = np.zeros((n_steps, n_samples))
    x = np.linspace(0, n_steps, n_steps) 
    y[0, :] = y0
    
    for i_sample in range(0, n_samples):
        i_step = 1
        while True:
            
            if i_step >= n_steps:
                break
            
            random_number = np.random.rand()
            u = (0.5 - random_number) * dx  # step length and direction
            if random_number >= normal_distribution(np.abs(u), D, dt):
                y[i_step, i_sample] = y[i_step-1, i_sample] + u

            i_step += 1       
    return x, y


def confined_diffusion(radius, n_steps, n_samples, dx, y0, D, dt):
    """
    Generates trajectories under confinement.

    :param radius: confinement radius
    :param n_steps: number of displacements
    :param n_samples: number of trajectories
    :param dx: displacement 
    :param y0: initial position
    :param D: diffusion coefficient
    :param dt: time step
    :return x, y: time, array containing N_samples trajectories with N_steps
    """
    y = np.zeros((n_steps, n_samples))
    x = np.linspace(0, n_steps, n_steps) 
    y[0, :] = y0
    sub_step = 0.0
    for i_sample in range(0, n_samples):  
       
        for i_step in range(0, n_steps):
            
            sub_x, sub_y = normal_diffusion(n_steps=100, n_samples=1, dx=dx, y0=sub_step, D=D, dt=dt)
            
            if sub_y[-1] < radius:
                t = i_step * dt
                y[i_step, i_sample] = sub_y[-1]
                sub_step = sub_y[-1]
                x[i_step] = t

    return x, y


def superdiffusion(velocity, n_steps, n_samples, y0, dt):
    """
    Generates direct diffusion trajectories. 
    Combine pairwise with normal diffusion components.
    
    :param velocity: constant velocity 
    :param n_steps: number of time steps
    :param n_samples: number of trajectories
    :param y0: initial position
    :param dt: time interval
    :return x, y: time, array containing N_samples trajectories with N_steps
    """
    y = np.zeros((n_steps, n_samples))
    x = np.linspace(0, n_steps, n_steps) 
    y[0, :] = y0
    
    for i_sample in range(0,n_samples):  
       
        for i_step in range(1, n_steps):
            y[i_step, i_sample] = y[i_step-1, i_sample] + velocity * dt
            x[i_step] = i_step * dt
            
    return x, y


def save_to_file(y, param, path):
    """
    Saves the trajectories to a file.

    :param y: trajectory array
    :param param: a parameter that characterizes the kind of trajectory
    :param path: path to the folder where the file will be saved
    """

    for n in range(0, len(y[:, 0])):
        np.savetxt(path + '/traj' + str(np.round(param, decimals=1)) + str(n) + '.csv', y[:, n],
                   delimiter=',', header='m')
