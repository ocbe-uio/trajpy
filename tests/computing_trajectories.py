import numpy as np
import trajpy.traj_generator as tjg
import trajpy.trajpy as tj
import unittest


class TestTrajectoryGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_anomalous_diffusion(self):
        """
        Test if the generator of anomalous trajectories is working properly
        by measuring the fractal dimension.
        """
        n_steps = 250
        n_samples = 1
        dt = 1.

        xa, trajectory = tjg.anomalous_diffusion(n_steps, n_samples, dt, alpha=1.0)

        r = tj.Trajectory()

        fractal_dimension, d_max = r.fractal_dimension_(trajectory)

        self.assertAlmostEqual(fractal_dimension, 3.08, places=1)

    def test_normal_diffusion(self):
        """
        Test if the generator of normal diffusion trajectories is working properly
        by measuring the fractal dimension.
        """
        n_steps = 250
        n_samples = 1
        dt = 0.1
        D = 100.0
        
        xa, trajectory = tjg.normal_diffusion(n_steps, n_samples, 1.0, 0., D, dt)
        
        r = tj.Trajectory()
        
        fractal_dimension, d_max = r.fractal_dimension_(trajectory)
        
        self.assertAlmostEqual(np.around(fractal_dimension, decimals=0), 2.0, places=1)
        
    def test_confined_diffusion(self):
        """
        Test if the generator of confined diffusion trajectories is working properly
        by measuring the fractal dimension.
        """
        n_steps = 250
        n_samples = 1
        dt = 0.1
        D = 100.0
        radius = 5.
        
        xa, trajectory = tjg.confined_diffusion(radius, n_steps, n_samples, 1.0, 0., D, dt)
        
        r = tj.Trajectory()
        
        fractal_dimension, d_max = r.fractal_dimension_(trajectory)
        
        self.assertAlmostEqual(np.around(fractal_dimension, decimals=0), 3.0, places=1)
        
    def test_superdiffusion(self):
        """
        Test if the generator of superdiffusion trajectories is working properly
        by measuring the fractal dimension.
        """
        n_steps = 250
        n_samples = 1
        dt = 0.1
        velocity = 1.0
        
        xa, trajectory = tjg.superdiffusion(velocity, n_steps, n_samples, 0., dt)
        
        r = tj.Trajectory()
        
        fractal_dimension, d_max = r.fractal_dimension_(trajectory)
        
        self.assertAlmostEqual(np.around(fractal_dimension, decimals=0), 1.0, places=1)  
        
if __name__ == '__main__':
    unittest.main()
