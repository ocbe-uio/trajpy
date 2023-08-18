import numpy as np
from typing import Union
import yaml

def parse_lammps_dump_yaml(filename):
    """
    Parse a LAMMPS dump file in YAML format to numpy array.
    The YAML file must be in the following format:
    ---
    time: 0.0
    natoms: 100
    keywords: [id, type, x, y, z, vx, vy, vz, fx, fy, fz]
    data:
    - [1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -nan, -nan, -nan]
    - [2, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -nan, -nan, -nan]
    - [3, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -nan, -nan, -nan]
    ...
    Parameters
    ----------
    filename : str
        Path to the LAMMPS dump file in YAML format.

    Returns
    -------
    positions : numpy.ndarray
        Array of shape (num_time_steps, num_atoms, 4) containing the
        positions of the atoms in the simulation box. The first column
        contains the time step, the second column contains the x
        coordinates, the third column contains the y coordinates and
        the fourth column contains the z coordinates.

    """
    with open(filename, 'r') as file:
        documents = list(yaml.load_all(file, Loader = yaml.FullLoader))
    
        # retrieving column names to assure consistency 
        # if the order of the columns changes in lammps 
        # yaml file standard
        keys = documents[0]["keywords"]
        column_dict = {key: index for index, key in enumerate(keys)}

        num_time_steps = len(documents)
        num_atoms = documents[0]['natoms']
        positions = np.zeros((num_time_steps, num_atoms, 4))

        for time_step, data in enumerate(documents):

            for atom_index, atom_properties in enumerate(data["data"]):

                positions[time_step, atom_index, 0] = data["time"]
                positions[time_step, atom_index, 1] = atom_properties[column_dict['x']]
                positions[time_step, atom_index, 2] = atom_properties[column_dict['y']]
                positions[time_step, atom_index, 3] = atom_properties[column_dict['z']]

    return positions


def moment_(trajectory: np.ndarray, order: int = 2, l_size: np.ndarray = np.array([0, 0]), periodic: bool = False) -> float:
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

def einstein_diffusion_probability(r: Union[float, np.ndarray], D: float, t: float) -> Union[float, np.ndarray]:
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


def unfold (r_old: np.ndarray, r: np.ndarray, box: Union[float, np.ndarray]) -> np.ndarray:
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
