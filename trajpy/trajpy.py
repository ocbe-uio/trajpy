import numpy as np
from scipy.stats import linregress
import trajpy.auxiliar_functions as aux
import warnings

class Trajectory(object):
    """
    This is the main class object in trajpy. It can be initialized
    as a dummy object for calling its functions or you can initialize
    it with a trajectory array or csv file.
    """
    def __init__(self, trajectory=np.zeros((1, 2)), box_length=None, **params):
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

        if box_length != None:
            self._r = aux.unfold(self._r[0], self._r, box_length)
            
        self.msd_ta = None  # time-averaged mean squared displacement
        self.msd_ea = None  # ensemble-averaged mean squared displacement
        self.msd_ratio = None
        self.anomalous_exponent = None
        self.fractal_dimension = None
        self.eigenvalues = None
        self.gyration_radius = None
        self.asymmetry = None
        self.straightness = None
        self.anisotropy = None
        self.kurtosis = None
        self.gaussianity = None
        self.msd_ratio = None
        self.efficiency = None
        self.confinement_probability = None
        self.diffusivity = None
        self._r0 = None  # maximum distance between any two points of the trajectory
        self.velocity = None
        self.velocity_description = None
        self.frequency_spectrum = None

    def compute_features(self):
        """
        Compute every feature for the trajectory saved in self._r.

        :return features: return the values of the features as a string.
        """
        self.msd_ta = self.msd_time_averaged(self._r,
                                             np.arange(len(self._r)))
        self.msd_ea = self.msd_ensemble_averaged(self._r)
        self.msd_ratio = self.msd_ratio_(self.msd_ta, n1=2, n2=10)
        self.anomalous_exponent = self.anomalous_exponent_(self.msd_ea, self._t)
        self.fractal_dimension, self._r0 = self.fractal_dimension_(self._r)

        self.gyration_radius = self.gyration_radius_(self._r)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.gyration_radius)
        self.eigenvalues[::-1].sort()  # the eigenvalues must be in the descending order
        self._idx = self.eigenvalues.argsort()[::-1]  # getting the position of the principal eigenvector
        self.kurtosis = self.kurtosis_(self._r, self.eigenvectors[self._idx[0]])
        self.anisotropy = self.anisotropy_(self.eigenvalues)

        self.straightness = self.straightness_(self._r)
        self.gaussianity = self.gaussianity_(self._r)
        self.efficiency = self.efficiency_(self._r)
        self.diffusivity = self.green_kubo_()
        self.velocity_description = self.velocity_description_(self.velocity)
        self.frequency_spectrum = self.frequency_spectrum_(self._r,self.t)
        #self.confinement_probability = self.confinement_probability_(2,self.diffusivity, self._t[-1])

        #(str(np.round(self.anomalous_exponent, 4)) + ',' +
        features = (str(np.round(self.msd_ratio, 4)) + ',' +
                    str(np.round(self.fractal_dimension, 4)) + ',' +
                    str(np.round(self.anisotropy, 4)) + ',' +
                    str(np.round(self.kurtosis, 4)) + ',' +
                    str(np.round(self.straightness, 4)) + ',' +
                    str(np.round(self.gaussianity, 4)) + ',' +
                    str(np.round(self.efficiency, 4)) + ',' +
                    str(np.round(self.diffusivity, 4)))

        return features


    @staticmethod
    def msd_time_averaged(spatial_components, tau):
        """
        calculates the time-averaged mean squared displacement
        
        .. math::
            \\langle \\mathbf{r}_{\\tau}^2 \\rangle = \\frac{1}{T-\\tau} \\sum_{t=1}^{N-\\tau} |\\mathbf{x}_{t+\\tau} - \\mathbf{x}_{\\tau} |^2

        where :math:`\\tau` is the time interval (time lag) between the two positions and :math:`T is total trajectory time length.

        :param spatial_components: array containing trajectory spatial coordinates
        :param tau: time lag, it can be a single value or an array
        :return msd: time-averaged MSD
        """
        if type(tau) == int:
            tau = np.asarray([tau])

        msd = np.zeros(len(tau))
        time_lag = 0
        for value in tau:

            dx = []

            for n in range(0, len(spatial_components) - value):
                dx.append(spatial_components[n + value] - spatial_components[n])

            dx = np.asarray(dx)

            msd[time_lag] = np.sum(np.power(dx, 2)) / (spatial_components.size - value + 1)
            time_lag += 1

        return msd

    @staticmethod
    def msd_ensemble_averaged(spatial_components):
        """
        calculates the ensemble-averaged mean squared displacement
        
        .. math::
            \\langle \\mathbf{r}^2 \\rangle (t) = \\frac{1}{N-1} \\sum_{n=1}^N |\\mathbf{x}_{n}-\\mathbf{x}_0|^2

        where :math:`N` is the number of trajectories,  :math:`\\mathbf{r}_n(t)` is the position of the trajectory :math:`n` at time :math:`t`.

        :param spatial_components: array containing trajectory spatial coordinates
        :return msd: ensemble-averaged msd
        """
        
        msd = np.zeros(len(spatial_components))
        for n in range(0, len(spatial_components)):
            msd[n] = np.sum(np.power(spatial_components[n] - spatial_components[0], 2))
        msd = msd / (len(spatial_components) - 1)

        return msd

    @staticmethod
    def msd_ratio_(msd_ta, n1, n2):
        """
        Ratio of the ensemble averaged mean squared displacements.

        .. math::
            \\langle r^2 \\rangle_{\\tau_1, \\tau_2} = \\frac{\\langle r^2 \\rangle_{\\tau_1 }}
            {\\langle r^2 \\rangle_{\\tau_2 }} - \\frac{\\tau_1}{\\tau_2}

        with
        
        .. math::
            \\tau_1 < \\tau_2
            
        :return msd_ratio:
        """

        msd_ratio = msd_ta[n1]/msd_ta[n2] - n1/n2
        return msd_ratio

    @staticmethod
    def anomalous_exponent_(msd, time_lag):
        """
        Calculates the diffusion anomalous exponent

        .. math::
            \\beta = \\frac{  \\partial \\log{ \\left( \\langle x^2 \\rangle   \\right)} }{ \\partial (\\log{(t)}) }

        :param msd: mean square displacement
        :param time_lag: time interval
        :return: diffusion nomalous exponent
        """

        msd_log = np.log(msd[1:])
        time_log = np.log(time_lag[1:])

        x, y = time_log, msd_log

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        anomalous_exponent = np.round(slope, decimals=2)

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

        return fractal_dimension, d_max

    @staticmethod
    def gyration_radius_(trajectory):
        """
        Calculates the gyration radius tensor of the trajectory

        .. math::
            R_{mn} =  \\frac{1}{2N^2} \\sum_{i=1}^N \\sum_{j=1}^N \\left( r_{m}^{(i)} -  r_{m}^{(j)}  \\right)\\left( r_{n}^{(i)} -  r_{n}^{(j)} \\right)\\, ,

        where :math:`N` is the number of segments of the trajectory, :math:`\\mathbf{r}_i` is the :math:`i`-th position vector along the trajectory,
        :math:`m` and :math:`n` assume the values of the corresponding coordinates along the directions :math:`x, y, z`.

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
    def asymmetry_(eigenvalues):
        """
        Takes the eigenvalues of the gyration radius tensor 
        to estimate the asymmetry between axis.

        .. math::
            a = - \\log{ \\left(1 - \\frac{ ( \\lambda_1 - \\lambda_2)^2}{2 ( \\lambda_1 + \\lambda_2)^2} \\right)}

        :param eigenvalues: eigenvalues of the gyration radius tensor
        :return: asymmetry coefficient
        """

        if len(eigenvalues) == 2:
            eigenvalues[::-1].sort() # the eigen values must the in the descending order
            
            asymmetry = - np.log(1. - np.power(eigenvalues[0] - eigenvalues[1], 2) /
                                 (2. * np.power(eigenvalues[0] + eigenvalues[1], 2)))
        else:
            raise IndexError("This function is meant for 2D trajectories only.")

        return asymmetry

    @staticmethod
    def anisotropy_(eigenvalues):
        """
        Calculates the trajectory anisotropy using the eigenvalues of the gyration radius tensor.

        .. math::
            a^2 = 1 - 3 \\frac{\\lambda_1\\lambda_2 + \\lambda_2 \\lambda_3 + \\lambda_3\\lambda_1 }{(\\lambda_1+\\lambda_2+\\lambda_3)^2}

        """
        
        eigenvalues[::-1].sort() # the eigen values must the in the descending order
        
        anisotropy = 1. - 3. * ((eigenvalues[0] * eigenvalues[1]
                                + eigenvalues[1] * eigenvalues[2]
                                + eigenvalues[2] * eigenvalues[0])
                                / np.power(np.sum(eigenvalues[:]), 2))

        return anisotropy

    @staticmethod
    def straightness_(trajectory):
        """
        Estimates how much straight is the trajectory

        .. math::
            S = \\frac{|\\mathbf{x}_{N-1} -\\mathbf{x}_0 |}
            { \\sum_{i=1}^{N-1} |\\mathbf{x}_i - \\mathbf{x}_{i-1}|}

        :return straightness: measure of linearity
        """
        summation = 0.

        for i_pos in range(1, len(trajectory)):
            summation += np.sqrt(np.dot(trajectory[i_pos] - trajectory[i_pos - 1],
                                        trajectory[i_pos] - trajectory[i_pos - 1]))

        straightness = np.sqrt(np.dot(trajectory[-1] - trajectory[0],
                                      trajectory[-1] - trajectory[0]))/summation
        return straightness

    @staticmethod
    def kurtosis_(trajectory, eigenvector):
        """
        We obtain the kurtosis by projecting each position of the trajectory along the main principal eigenvector of the radius of gyration tensor
         :math:`r_i^p = \\mathbf{r} \\cdot \\hat{e}_1` and then calculating the quartic moment

        .. math::
            K = \\frac{1}{N} \\sum_{i=1}^N \\frac{ \\left(r_i^p - \\langle r^p \\rangle \\right)^4}{\\sigma_{r^p}^4} \\, ,

        where :math:`\\langle r^p \\rangle` is the mean position of the projected trajectory and :math:`\\sigma_{r^p}^2` is the variance.
        The kurtosis measures the peakiness of the distribution of points in the trajectory.

        :return kurtosis: K
        """
        N = len(trajectory)
        r_projection = np.zeros(N)
        for n, position in enumerate(trajectory):
            r_projection[n] = np.dot(position, eigenvector)

        mean_ = r_projection.mean()
        std_ = r_projection.std()
        r_projection -= mean_
        kurtosis = (1./N) * np.sum(np.power(r_projection,4))/np.power(std_, 4)


        return kurtosis

    @staticmethod
    def gaussianity_(trajectory):
        """
        measure of how close to a gaussian distribution is the trajectory.

        .. math::
            g(n) = \\frac{ \\langle r_n^4 \\rangle }{2 \\langle r_n^2 \\rangle^2}

        :return gaussianity: measure of similarity to a gaussian function
        """
        fourth_order = aux.moment_(trajectory, 4)
        second_order = aux.moment_(trajectory, 2)

        gaussianity = (2/3) * (fourth_order/second_order) - 1

        return gaussianity

    @staticmethod
    def confinement_probability_(r0, D, t, N=100):
        """ new
        Estimate the probability of Brownian particle with
        diffusivity :math:`D` being trapped in the interval :math:`[-r0, +r0]` after a period of time t.
        
        .. math::
            P(r, D, t) = \\int_{-r_0}^{r_0} p(r, D, t) \\mathrm{d}r

        :param r: position
        :param D: diffusivity
        :param t: time length
        :return probability: probability of the particle being confined
        """
        p = np.zeros(N)
        X = np.linspace(-r0, r0, N)
        dx = X[1]-X[0]
        for n, x in enumerate(X):
            p[n] = aux.einstein_diffusion_probability(x, D, t)
        probability = np.sum(p)*dx
        return 1-probability

    @staticmethod
    def efficiency_(trajectory):
        """
            Calculates the efficiency of the movement, a measure that is related to
            the straightness.

        .. math::
            E = \\frac{|\\mathbf{x}_{N-1} - \\mathbf{x}_{0}|^2  }
            { (N-1) \\sum_{i=1}^{N-1} |\\mathbf{x}_{i} - \\mathbf{x}_{i-1}|^2 }

        :return efficiency: trajectory efficiency.
        """
        den = 0.

        for n in range(1, len(trajectory)):
            den += np.sum(np.power(trajectory[n] - trajectory[n - 1], 2))

        efficiency = np.sum(np.power(trajectory[-1] - trajectory[0], 2)) / \
            ((len(trajectory) - 1) * den)

        return efficiency
    
    @staticmethod
    def velocity_(position,time):
        """
            Computes the velocity associated with the trajectory
        """
        
        velocity = np.diff(position, axis=0)/(time[1]-time[0])

        return velocity

    
    def _stationary_velocity_correlation(self, taus):
        """
            Computes the stationary velocity autocorrelation function by time average

            .. math:
                \\langle \\vec{v(t+\\tau)} \\vec{v(t)} \\rangle
        
        :param taus: single or array of non-negative integer values representing the time lag
        :return time_averaged_corr_velocity: velocity autocorrelation function output
        """

        time_averaged_corr_velocity = np.zeros(len(taus))
        N = len(self.velocity)
        for tau in taus:
            time_averaged_corr_velocity[tau] = (np.sum(np.einsum('ij,ij->i',
                np.take(a=self.velocity,indices=np.arange(0,N-tau)+tau,axis=0),
                np.take(a=self.velocity,indices=np.arange(0,N-tau),axis=0)))+ 
                time_averaged_corr_velocity[tau-1])*(self._t[1]-self._t[0]) / (N-tau)
        return time_averaged_corr_velocity

    
    def green_kubo_(self):
        """
            Computes the generalised Green-Kubo's diffusion constant

            :return diffusivity: diffusion constant obtained by the Gree-Kubo relation 
        """
        
        self.velocity_()
        self.diffusivity = 0.
        N = len(self.velocity)
        dt = self._t[1] - self._t[0]

        for tau in range(1, N-1):
            self.diffusivity += self._stationary_velocity_correlation(tau) * dt/3

        return self.diffusivity
    
    @staticmethod
    def velocity_description_(velocity):     
        """
            Computes the main features of the velocity distribuition: mean, median, mode, variance,
            standard deviation, range, skewness and kurtosis
        
            return velocity_description_: returns a dictionary where the values are bounded 
            to a key of the same name
        """
        mean = np.mean(velocity,axis=0)  
        median = np.median(velocity,axis=0)
        standard_deviation = np.std(velocity,axis=0) 
        variance = np.var(velocity,axis=0)
        ran = np.abs(np.max(velocity,axis=0) - np.min(velocity,axis=0))
        skewness = (sum((velocity[:,:]-mean)**3)/len(velocity))/((sum((velocity[:,:]-mean)**2)/len(velocity))**1.5)
        kurtosis = (sum((velocity[:,:]-mean)**4)/len(velocity))/(standard_deviation**4) - 3
        mode = np.empty(velocity.shape[1],dtype=object) 
        for col in range(velocity.shape[1]):
            vel_values, vel_freq = np.unique(np.round(velocity[:,col],3), return_counts=True,axis=0)
        if max(vel_freq)==1:
            mode[col] = 'no mode'
        else:
            mode[col] = vel_values[np.where(vel_freq==max(vel_freq))]
        
        velocity_description = {'mean':mean,
                'median':median,
                'mode':mode,
                'standard_deviation':standard_deviation,
                'variance':variance,
                'range':ran,
                'kurtosis':kurtosis,
                'skewness':skewness}
        return velocity_description
    
    @staticmethod
    def frequency_spectrum_(position,t): 
        """
            Computes the Fourier Transform of the trajectory and calculates the dominant frequency,
            dominant amplitude, main frequencies (given a threshold of 1 Hz), the corresponing
            amplitudes and mean frequency

            return frequency_spectrum: returns a dictionary where the values are bounded 
            to a key of the same name
        """   
        dt = t[1] - t[0]
        n = len(t)
        yvalues = position
        xvalues = range(len(yvalues))
        yvalues_detrended = np.zeros(shape=yvalues.shape)
        for col in range(yvalues.shape[1]):
            z1 = np.polyfit(xvalues, yvalues[:,col], deg=1)
            p1 = np.poly1d(z1)
            yvalues_detrended[:,col] = yvalues[:,col] - p1(xvalues) 
        fourier = np.fft.fft(yvalues_detrended,axis=0,n=n) 
        limite = np.arange(1,np.floor(n/2),dtype='int') 
        poder = 2*np.abs(fourier)/n 
        poder = poder[1:max(limite),:] 
        f = (1/(dt*n))*np.arange(n) 
        f = f[1:max(limite)] 
        dominant_frequency = f[np.argmax(poder[:max(limite)],axis=0)] 
        dominant_amp = np.max(poder[:max(limite)],axis=0) 
        main_frequencies = np.empty(shape=yvalues_detrended.shape[1],dtype=object) 
        main_amplitudes = np.empty(shape=yvalues_detrended.shape[1],dtype=object)
        mean_frequency = np.zeros(yvalues_detrended.shape[1])
        for col in range(yvalues_detrended.shape[1]):
            indice = np.where(poder[:,col]>=1)
            main_amplitudes[col] = poder[indice,col] 
            main_frequencies[col] = f[indice] 
            mean_frequency[col] = np.mean(indice) 
        frequency_spectrum = {'dominant frequency':dominant_frequency,
                            'dominant amplitude':dominant_amp,
                            'mean frequency':mean_frequency,
                            'main frequencies':main_frequencies,
                            'main amplitudes':main_amplitudes,
                            'x':f,
                            'y':poder}
        return frequency_spectrum