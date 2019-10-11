import numpy as np
from scipy import signal


def moment_(trajectory, order=2, l_size=np.array([0, 0]), periodic=False):
    """
        this function calculates the nth moment of the trajectory r

    :param trajectory: trajectory
    :param l_size: box size
    :param periodic: boundary conditions
    :param order: momentum order
    :return:
    """

    moment = np.zeros(trajectory.shape)
    n_points = len(trajectory)

    for n in range(0, n_points - 1):

        dr = trajectory[n + 1] - trajectory[n]

        if periodic:
            dr = dr - np.round(dr / l_size, decimals=0) * l_size

        moment[n] = np.sum(np.power(dr, order))
    return np.sum(moment) / (n_points - 1)


class Trajectory(object):

    def __init__(self, trajectory=np.zeros((1, 2)), compute_all=False, **params):

        if type(trajectory) == np.ndarray:
            self._t, self._r = trajectory[:, 0], trajectory[:, 1:]
        elif type(trajectory) == tuple:
            self._t, self._r = np.asarray(trajectory[0]), np.asarray(trajectory[1:])
        elif type(trajectory) == str:
            trajectory = np.genfromtxt(trajectory, **params)
            self._t, self._r = trajectory[:, 0], trajectory[:, 1:]
        else:
            raise TypeError('trajectory receives an array or a filename as input.')

        if compute_all:
            self.msd = self.mean_squared_displacement(self._r)
            self.fractal_dimension = self.fractal_dimension_(self._r)
            self.gyration_radius = self.gyration_radius_(self._r)
            self.asymmetry = self.asymmetry_(self.gyration_radius)
            self.straightness = self.straightness_(self._r)
            self.kurtosis = self.kurtosis_(self._r)
            self.gaussianity = self.gaussianity_(self._r)
            self.msd_ratio = self.msd_ratio_(self._r)
            self.efficiency = self.efficiency_(self._r)

    @staticmethod
    def ensemble_averaged_msd(trajectory):
        """
            calculates the ensemble-averaged mean squared displacement
            $\langle \mathbf{r}_n^2 \rangle = \frac{1}{N-n} \sum_{n=1}^{N-n} |\mathbf{x}_{i+n} - \mathbf{x}_n |^2$
            $n = 1, \ldots, N-1$

        """

        return msd

    @staticmethod
    def time_averaged_msd(trajectory):
        """
            calculates the time-averaged mean squared displacement
            $\langle \mathbf{r}_n^2 \rangle (t) = sum_n^N |\mathbf{x}_{n}-\mathbf{x}_0|**2$
        """
        msd = np.zeros(len(trajectory))
        for n in range(0, len(trajectory)):
            msd[n] = np.sum(np.power(trajectory[n] - trajectory[0], 2))
        msd = signal.savgol_filter(msd, 3, 1, mode='nearest') / (len(trajectory) - 1)

        return msd

    @staticmethod
    def anomalous_exponent_(mean_squared_displacement):
        """
            calculates the anomalous exponent
        :param mean_squared_displacement:
        :return:
        """
        anomalous_exponent = np.mean(mean_squared_displacement)  # placeholder

        return anomalous_exponent

    @staticmethod
    def fractal_dimension_(trajectory):
        """
        :return fractal_dimension: calculates the fractal dimension
                                    log(N)/(log(dNL**-1)
        """
        dr = np.zeros(np.power(len(trajectory), 2))

        # calculating the distance between each pair of points in the trajectory
        n_distance = 0
        for i_pos in range(0, len(trajectory) - 1):
            for j_pos in range(i_pos + 1, len(trajectory) - 1):
                dr[n_distance] = np.sum(np.power(trajectory[i_pos] - trajectory[j_pos], 2))
                n_distance += 1

        d_max = np.sqrt(np.max(dr))  # maximum distance between any two points of the trajectory
        n_points = trajectory.size
        length = 0
        diff = np.zeros(trajectory.shape)

        for i_pos in range(0, len(trajectory) - 1):
            diff[i_pos] = np.round(trajectory[i_pos + 1], decimals=2) \
                          - np.round(trajectory[i_pos], decimals=2)
            length += np.sqrt(np.sum(np.power(diff[i_pos], 2)))

        fractal_dimension = np.round(np.log(n_points) / (np.log(n_points)
                                     + np.log(d_max * np.power(length, -1))), decimals=2)

        return fractal_dimension

    @staticmethod
    def gyration_radius_(trajectory):
        """
            calculates the gyration radius tensor of the trajectory

        :return gyration_radius:
        """

        dim = trajectory.shape[1]  # number of dimensions
        r_gyr = np.zeros((dim, dim))  # gyration radius tensor
        r_mean = np.mean(trajectory, axis=0)

        for m in range(0, dim):
            for n in range(0, dim):
                r_gyr[m, n] = np.sum(np.matmul(trajectory[:, m] - r_mean[m],
                                               trajectory[:, n] - r_mean[n]))

        gyration_radius = r_gyr/trajectory.size

        return gyration_radius

    @staticmethod
    def asymmetry_(gyration_radius):
        """
            takes the gyration radius as input and calculates the eigenvalues
            then use the eigenvalues to estimate the asymmetry between axis

        :param gyration_radius:
        :return:
        """

        eigen_values = np.linalg.eigvals(gyration_radius)

        asymmetry = - np.log(1. - np.power(eigen_values[0] - eigen_values[1], 2) /
                             (2. * np.power(eigen_values[0] + eigen_values[1], 2)))

        return asymmetry

    @staticmethod
    def straightness_(trajectory):
        """
            estimates how much straight is the trajectory
        :param trajectory:
        :return:
        """
        summation = 0.

        for i_pos in range(1, len(trajectory)-1):
            summation += np.sqrt(np.dot(trajectory[i_pos]-trajectory[i_pos-1],
                                        trajectory[i_pos]-trajectory[i_pos-1]))
        straightness = np.sqrt(np.dot(trajectory[-1]-trajectory[0],
                                      trajectory[1-1]-trajectory[0]))/summation
        return straightness

    @staticmethod
    def kurtosis_(trajectory):
        """
            calculates the kurtosis of the trajectory projecting the positions
            along the principal axis calculated with the gyration radius
        :param trajectory:
        :return:
        """

        kurtosis = np.mean(trajectory)  # placeholder

        return kurtosis

    @staticmethod
    def gaussianity_(trajectory):
        """
            measure of how close to a gaussian distribution is the trajectory
        :param trajectory:
        :return:
        """
        fourth_order = moment_(trajectory, 4)
        second_order = moment_(trajectory, 2)

        gaussianity = (2/3) * (fourth_order/second_order) - 1

        return gaussianity

    @staticmethod
    def msd_ratio_(trajectory):
        """
            ratio of mean squared displacements
        :param trajectory:
        :return:
        """
        msd_ratio = np.mean(trajectory)  # placeHolder
        return msd_ratio

    @staticmethod
    def trappedness_(diffusion_constant, r0):
        """
            estimate the trappedness probability
        :param diffusion_constant:
        :param r0:
        :return:
        """
        trappedness = 1 - np.exp(0.2080 - 0.25117 * (diffusion_constant / np.power(r0, 2)))
        return trappedness

    @staticmethod
    def efficiency_(trajectory):
        """
            calculates the efficiency of the movement
        :param trajectory:
        :return:
        """
        den = 0.

        for n in range(1, len(trajectory)):
            den += np.sum(np.power(trajectory[n] - trajectory[n - 1], 2))

        efficiency = np.sum(np.power(trajectory[-1] - trajectory[0], 2)) / \
            ((len(trajectory) - 1) * den)

        return efficiency


if __name__ == '__main__':

    N = 100
    Length = 100
    t = np.linspace(0, N, N)
    #  r = np.random.rand(Length, 2)
    r = np.zeros((Length, 2))
    for i in range(0, Length):
        r[i] = np.array([i, i])
    t = Trajectory(r)

    """
    import matplotlib.pyplot as plt
    r = []
    for i in range(0, N):
        r.append(np.random.rand(Length,3))
    r[:,0] = t
    msd = np.zeros(Length)
    for traj in r:
        msd += Trajectory.mean_squared_displacement(traj)
    msd = msd/N
    a = Trajectory(r, compute_all=True)
    plt.plot(msd)
    plt.show()
    """
