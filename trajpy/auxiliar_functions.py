import numpy as np

def moment_(trajectory, order=2, l_size=np.array([0, 0]), periodic=False):
    """
    Calculates the n-th statistical moment of the trajectory r.

    .. math::
        \\langle \\mathbf{r}_n \\rangle = \\frac{1}{N-1} \\sum_{i=1}^{N-1} |\\mathbf{x}_{i+1} - \\mathbf{x}_i |^n

    :param trajectory: trajectory
    :param l_size: box size
    :param periodic: boundary conditions
    :param order: momentum order
    :return moment: n-th moment
    """

    moment = np.zeros(trajectory.shape)
    n_points = len(trajectory)

    for n in range(0, n_points - 1):

        dr = trajectory[n + 1] - trajectory[n]

        if periodic:
            dr = dr - np.round(dr / l_size, decimals=0) * l_size

        moment[n] = np.sum(np.power(dr, order))
    return np.sum(moment) / (n_points - 1)

def einstein_diffusion_probability(r, D, t):
    """
    Calculates the probability of a Brownian particle with
    diffusivity D arriving in the position r after a period of time t.

    The normalized probability is given by
    .. math::
        p(r, t) = \\frac{1}{\\sqrt{ 4 \\pi D t}} \\exp{  \\left( \\frac{r^2}{4Dt} \\right)} \\right) \\, .

    :param r: position
    :param D: diffusivity
    :param t: time length
    :return probability: probability of arriving in r.
    """
    A = 1. / np.power(4. * np.pi * D * t, 0.5)
    probability = A * np.exp(-np.power(r, 2) / (4. * D * t))
    return probability


def unfold (r_old, r, box):
    """
    Removes effects of periodic boundaries on particle trajectories.
    r_old is the configuration at the previous step 
    r is the current configuration
    box is accessed from the calling program.
    The function returns the unfolded version of r.
    
    From the book: Computer Simulation of Liquids
    git: github.com/Allen-Tildesley/
    Authors: Michael P. Allen and Dominic J. Tildesley
    """

    r_new = r - r_old                      #  Convert r to displacements relative to r_old
    r_new = r_new - np.rint(r_new/box)*box # Apply periodic boundaries to displacements
    r_new = r_new + r_old                  # Convert r back to absolute coordinates
    return r_new
