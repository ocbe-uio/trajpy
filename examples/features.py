import numpy as np
import matplotlib.pyplot as plt
import trajpy.trajpy as tj

filename = '../data/samples/sample.csv'
r = tj.Trajectory(filename, skip_header=1, delimiter=',')

r.compute_features()

print(r.efficiency,
      r.asymmetry,
      r.gaussianity,
      r.straightness,
      r.gyration_radius,
      )
"""
self.msd = self.time_averaged_msd(self._r)
self.fractal_dimension = self.fractal_dimension_(self._r)
self.gyration_radius = self.gyration_radius_(self._r)
self.asymmetry = self.asymmetry_(self.gyration_radius)
self.straightness = self.straightness_(self._r)
self.kurtosis = self.kurtosis_(self._r)
self.gaussianity = self.gaussianity_(self._r)
self.msd_ratio = self.msd_ratio_(self._r)
self.efficiency = self.efficiency_(self._r)
"""


#  msd = mean_squared_displacement(r)
#  plt.plot(r._t, r._r)
#  plt.show()
