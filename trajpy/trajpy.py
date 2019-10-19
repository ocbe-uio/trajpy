import numpy as np
from sklearn.linear_model import LinearRegression


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
    """
    This is the main class object in trajpy. It can be initialized
    as a dummy object for calling its functions or you can initialize
    it with a trajectory array or csv file.
    """
    def __init__(self, trajectory=np.zeros((1, 2)), **params):
        """
        Initialization function that can be left blank for using staticmethods.
        It can be initialized with an array with shape (N, dim)
        where dim is the number of spatial dimensions plus the time component.
        The first column must be the time, followed by the x- and y-axis.
        It also accepts tuples (t, x, y) or csv files.

        The trajectory will be split between the temporal component self._t
        and the spatial axis self._r.

        :param trajectory: 2D trajectory as a function of time (t, x, y)
        :param params: use params for passing parameters into np.genfromtxt()
        """
        if type(trajectory) == str:
            trajectory = np.genfromtxt(trajectory, **params)

        if type(trajectory) == np.ndarray:
            self._t, self._r = trajectory[:, 0], trajectory[:, 1:]
        elif type(trajectory) == tuple:
            self._t, self._r = np.asarray(trajectory[0]), np.asarray(trajectory[1:])
        else:
            raise TypeError('trajectory receives an array or a filename as input.')

        self.msd_ta = None
        self.msd_ea = None
        self.anomalous_exponent = None
        self.fractal_dimension = None
        self.gyration_radius = None
        self.asymmetry = None
        self.straightness = None
        self.kurtosis = None
        self.gaussianity = None
        self.msd_ratio = None
        self.efficiency = None

    def compute_features(self):
        """
            compute every feature for the trajectory saved in self._r
        """
        # self.msd_ta = self.msd_time_averaged(self._r)
        self.msd_ea = self.msd_ensemble_averaged(self._r,
                                                 np.arange(len(self._r)))
        self.anomalous_exponent = self.anomalous_exponent_(self.msd_ea, self._t)
        self.fractal_dimension = self.fractal_dimension_(self._r)
        self.gyration_radius = self.gyration_radius_(self._r)
        self.asymmetry = self.asymmetry_(self.gyration_radius)
        self.straightness = self.straightness_(self._r)
        # self.kurtosis = self.kurtosis_(self._r)
        self.gaussianity = self.gaussianity_(self._r)
        # self.msd_ratio = self.msd_ratio_(self._r)
        self.efficiency = self.efficiency_(self._r)

        features = (str(np.round(self.anomalous_exponent, 4)) + ',' +
                    str(np.round(self.fractal_dimension, 4)) + ',' +
                    str(np.round(self.asymmetry, 4)) + ',' +
                    str(np.round(self.straightness, 4)) + ',' +
                    str(np.round(self.gaussianity, 4)) + ',' +
                    str(np.round(self.efficiency, 4)))
        return features

    @staticmethod
    def msd_ensemble_averaged(trajectory, tau):
        """
        calculates the ensemble-averaged mean squared displacement
        
        .. math::
            \\langle \\mathbf{r}_n^2 \\rangle = \\frac{1}{N-n} \\sum_{n=1}^{N-n} |\\mathbf{x}_{i+n} - \\mathbf{x}_n |^2
        .. math::
            n = 1, \\ldots, N-1
        :param trajectory: trajectory array
        :param tau: time lag, it can be a single value or an array
        :return msd: return the ensemble averaged mean square displacement
        """
        if type(tau) == int:
            tau = np.asarray([tau])

        msd = np.zeros(len(tau))
        time_lag = 0
        for value in tau:

            dx = []

            for n in range(0, len(trajectory) - value):
                dx.append(trajectory[n + value] - trajectory[n])

            dx = np.asarray(dx)

            msd[time_lag] = np.sum(np.power(dx, 2)) / (trajectory.size - value + 1)
            time_lag += 1

        return msd

    @staticmethod
    def msd_time_averaged(trajectory):
        """
        calculates the time-averaged mean squared displacement
        
        .. math::
            \\langle \\mathbf{r}_n^2 \\rangle (t) = \\sum_n^N |\\mathbf{x}_{n}-\\mathbf{x}_0|^2
        :return msd: time-averaged msd
        """
        msd = np.zeros(len(trajectory))
        for n in range(0, len(trajectory)):
            msd[n] = np.sum(np.power(trajectory[n] - trajectory[0], 2))
        msd = msd / (len(trajectory) - 1)

        return msd

    @staticmethod
    def anomalous_exponent_(msd, timelag):
        """
        Calculates the diffusion anomalous exponent

        .. math::
            \\alpha = \\frac{ \log{\\left( \\langle x^2 \\rangle   \\right)} }{ \log{(t)} }

        :param msd: mean square displacement
        :param timelag: time interval
        :return: diffusion nomalous exponent
        """

        msd_log = np.log(msd[1:])
        time_log = np.log(timelag[1:])

        x, y = time_log, msd_log
        x = x.reshape(-1, 1)

        reg = LinearRegression()
        reg.fit(x, y)

        anomalous_exponent = np.round(reg.coef_[0], decimals=2)

        return anomalous_exponent

    @staticmethod
    def fractal_dimension_(trajectory):
        """
        Estimates the fractal dimension of the trajectory

        .. math::
                 \\frac{\\log{(N)} }{ \\log{(dNL^{-1})}}

        :return fractal_dimension: returns the fractal dimension
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
        Calculates the gyration radius tensor of the trajectory

        :return gyration_radius: tensor
        """

        dim = trajectory.shape[1]  # number of dimensions
        r_gyr = np.zeros((dim, dim))  # gyration radius tensor
        r_mean = np.mean(trajectory, axis=0)

        for m in range(0, dim):
            for n in range(0, dim):
                r_gyr[m, n] = np.sum(np.matmul(trajectory[:, m] - r_mean[m],
                                               trajectory[:, n] - r_mean[n]))

        gyration_radius = np.sqrt(np.abs(r_gyr/trajectory.size))

        return gyration_radius

    @staticmethod
    def asymmetry_(gyration_radius):
        """
        Takes the gyration radius as input and calculates the eigenvalues
        then use the eigenvalues to estimate the asymmetry between axis

        .. math::
            a = - \\log{ \\left(1 - \\frac{ ( \\lambda_1 - \\lambda_2)^2}{2 ( \\lambda_1 + \\lambda_2)^2} \\right)}

        TODO: generalize for 1D and 3D.

        :param gyration_radius: gyration radius tensor
        :return: asymmetry coefficient
        """

        eigen_values = np.linalg.eigvals(gyration_radius)

        # this formula is specific for a two dimensional system
        asymmetry = - np.log(1. - np.power(eigen_values[0] - eigen_values[1], 2) /
                             (2. * np.power(eigen_values[0] + eigen_values[1], 2)))

        return asymmetry

    @staticmethod
    def straightness_(trajectory):
        """
        Estimates how much straight is the trajectory
        .. math::
            S = \\frac{|\\mathbf{x}_{N-1} -\\mathbf{x}_0 |}{ \\sum_{i=1}{N-1} |\\mathbf{x}_i - \\mathbf{x}_{i-1}|}

        :return straightness: measure of linearity
        """
        summation = 0.

        for i_pos in range(1, len(trajectory) - 1):
            summation += np.sqrt(np.dot(trajectory[i_pos] - trajectory[i_pos - 1],
                                        trajectory[i_pos] - trajectory[i_pos - 1]))

        straightness = np.sqrt(np.dot(trajectory[-1] - trajectory[0],
                                      trajectory[-1] - trajectory[0]))/summation
        return straightness

    @staticmethod
    def kurtosis_(trajectory):
        """
        Calculates the kurtosis of the trajectory projecting the positions
        along the principal axis calculated with the gyration radius
        .. math::
            K = \\frac{1}{N} \\sum_{i=1}^N \\frac{x_i^p - \\bar{x}_i^p }{ \\sigma_{x^p}^4}

        :return kurtosis:
        """

        kurtosis = np.mean(trajectory)  # placeholder

        return kurtosis

    @staticmethod
    def gaussianity_(trajectory):
        """
        measure of how close to a gaussian distribution is the trajectory.

        .. math::
            g(n) = \\frac{ \\langle r_n^4 \\rangle }{2 \\langle r_n^2 \\rangle^2}

        :return gaussianity: measure of similarity to a gaussian function
        """
        fourth_order = moment_(trajectory, 4)
        second_order = moment_(trajectory, 2)

        gaussianity = (2/3) * (fourth_order/second_order) - 1

        return gaussianity

    @staticmethod
    def msd_ratio_(trajectory):
        """
        ratio of mean squared displacements

        .. math::
            \\langle r^2 \\rangle_{n_1, n_2} = \\frac{\\langle r^2 \\rangle_{n_1 }}
            {\\langle r^2 \\rangle_{n_2 }} - \\frac{n_1}{n_2}

        :return msd_ratio:
        """
        msd_ratio = np.mean(trajectory)  # placeHolder
        return msd_ratio

    @staticmethod
    def trappedness_(diffusion_constant, r0):
        """
        Estimates the trappedness probability:

        .. math::
        p_t = 1 - \\exp{ \\left( 0.2048  - 0.25117 \\left( \\frac{Dt}{r_0^2} \\right) \\right)}

        :param diffusion_constant: short-time diffusion coefficient estimated by the first two time lags
        :param r0: radius of the bounded region
        :return:
        """
        trappedness = 1 - np.exp(0.2080 - 0.25117 * (diffusion_constant / np.power(r0, 2)))
        return trappedness

    @staticmethod
    def efficiency_(trajectory):
        """
            calculates the efficiency of the movement
        :return efficiency:
        """
        den = 0.

        for n in range(1, len(trajectory)):
            den += np.sum(np.power(trajectory[n] - trajectory[n - 1], 2))

        efficiency = np.sum(np.power(trajectory[-1] - trajectory[0], 2)) / \
            ((len(trajectory) - 1) * den)

        return efficiency
