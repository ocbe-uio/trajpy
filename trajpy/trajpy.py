import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.linear_model import LinearRegression


class features(object):

    def __init__(self, filename, plot=False):
        self.r = np.genfromtxt(filename, skip_header=1)
        self.shape = self.r.shape
        self.size = self.r[:, 0].size
        self.MSD = self.mean_squared_displacement()
        self.dimension = self.fractal_dimension()
        self.exponent = self.anomalous_exponent(plot)

    def mean_squared_displacement(self):

        self.MSD = np.zeros(self.shape)

        for n in range(0, self.size):
            try:
                self.MSD[n, :] = np.sum(np.power(self.r[n, :] - self.r[0, :], 2))
            except:
                self.MSD[n] = np.sum(np.power(self.r[n, :] - self.r[0, :], 2))

        self.MSD = signal.savgol_filter(self.MSD, 3, 1, mode='nearest')

        return self.MSD# /np.std(m)



    def fractal_dimension(self):

        dr = np.zeros(np.power(self.size, 2))

        n = 0
        for i in range(0, self.size - 1):
            for j in range(i + 1, self.size - 1):
                dr[n] = np.sum(np.power(self.r[i, :] - self.r[j, :], 2))
                n += 1

        d = np.sqrt(np.max(dr))
        N = self.size
        L = 0
        diff = np.zeros(self.shape)


        for i in range(0, self.size - 1):
            diff[i,:] = np.round(self.r[i + 1,:], decimals=2) \
                     - np.round(self.r[i,:], decimals=2)
            L += np.sqrt(np.sum(np.power(diff[i,:], 2)))

        self.dimension = np.round(np.log(N) / (np.log(N) + np.log(d * np.power(L, -1))), decimals=2)
        return self.dimension


    def anomalous_exponent(self, plot=False):

        if plot:
            plt.plot(self.r[:,0], self.MSD)
            plt.show()

        MSD_log = np.log(self.MSD[1:])
        time_log = np.log(self.r[1:,0])

        X, Y = time_log, MSD_log
        X = X.reshape(-1, 1)

        reg = LinearRegression()
        X_test = X
        X_r = X
        reg.fit(X, Y)
        X_test = X_test.reshape(-1, 1)
        Y_pred = reg.predict(X_test)

        if plot:
            plt.plot(X_r, Y)
            plt.plot(X_test, Y_pred)
            plt.show()

        return np.round(reg.coef_[0], decimals=2), np.round(reg.score(X,Y),decimals=4)