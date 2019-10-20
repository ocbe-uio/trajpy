import numpy as np
import trajpy.traj_generator as tjg
import trajpy.trajpy as tj
import unittest


class TestTrajectoryGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_anomalous_trajectory(self):
        """
            test if the generator of anomalous trajectories is working properly
            by measuring the fractal dimension
        """
        tsteps = 250
        nsample = 1
        dt = 1.

        xa, trajectory = tjg.anomalous_diffusion(tsteps, nsample, dt, alpha=1.0)

        r = tj.Trajectory()

        fractal_dimension, d_max = r.fractal_dimension_(trajectory)

        self.assertAlmostEqual(fractal_dimension, 3.08, places=1)


if __name__ == '__main__':
    unittest.main()
