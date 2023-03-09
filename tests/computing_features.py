import numpy as np
import unittest

import os
src_path = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/trajpy/')
sys.path.append(src_path)
from trajpy import trajpy as tj

class TestFeatures(unittest.TestCase):
 
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_efficiency(self):
        """
            testing if the efficiency is being computed correctly for three cases
            1) linear trajectory 2) random steps and 3) oscillatory.
        """

        x1 = np.linspace(0, 100, 100)
        x2 = np.random.rand(100)
        x3 = np.cos(np.pi * np.linspace(0, 100, 1000))

        r = tj.Trajectory()

        self.assertAlmostEqual(r.efficiency_(x1), 1.0, places=2)
        self.assertAlmostEqual(r.efficiency_(x2), 0.0, places=2)
        self.assertAlmostEqual(r.straightness_(x3), 0.0, places=1)

    def test_fractal_dimension(self):
        """
            testing if the fractal dimension is being computed correctly
        """

        x1 = np.linspace(0, 100, 100)
        x2 = np.random.rand(100)

        r = tj.Trajectory()

        self.assertAlmostEqual(r.fractal_dimension_(x1)[0], 1.0, places=2)
        self.assertGreaterEqual(r.fractal_dimension_(x2)[0], 3.0)

    def test_gyration_radius_and_asymmetry(self):
        """
            testing the radius of gyration for some basic cases
            circle: Rg = R/2
        """
        length = 20
        circle = np.zeros((length, 2))
        radius = 10
        circle_gyr = np.array([[radius, 0.], [0., radius]])/2

        x1 = np.linspace(0, 100, 1000)
        x2 = np.cos(np.pi * np.linspace(0, 100, 1000))
        x1x2 = np.array([x1, x2]).transpose()

        for n in range(0, length):
            circle[n] = radius * np.array([np.cos(np.pi * n/10), np.sin(np.pi * n/10)])

        r = tj.Trajectory()
        gyration_radius_circle = r.gyration_radius_(circle)
        gyration_radius_osci = r.gyration_radius_(x1x2)
        eigenvalues_circle = np.linalg.eigvals(gyration_radius_circle)
        eigenvalues_osci = np.linalg.eigvals(gyration_radius_osci)
        asymmetry_oscillatory = r.asymmetry_(eigenvalues_osci)
        asymmetry_circle = r.asymmetry_(eigenvalues_circle)

        self.assertAlmostEqual(asymmetry_oscillatory, asymmetry_oscillatory, places=1)
        self.assertAlmostEqual(asymmetry_circle, asymmetry_circle, places=1)
        self.assertEqual(np.round(gyration_radius_circle, 2).all(), np.round(circle_gyr, 2).all())


    def test_straightness(self):
        """
            testing the straightness function for three cases
            x1 - linear; x2 - oscillatory; x1x2 - mixed components
        """

        x1 = np.linspace(0, 100, 1000)
        x2 = np.cos(np.pi * np.linspace(0, 100, 1000))
        x1x2 = np.array([x1, x2]).transpose()

        r = tj.Trajectory()

        self.assertAlmostEqual(r.straightness_(x1), 1.0, places=2)
        self.assertAlmostEqual(r.straightness_(x2), 0.0, places=2)
        self.assertAlmostEqual(r.straightness_(x1x2), 0.43, places=1)

    def test_kurtosis(self):
        """
            testing the kurtosis function
        """
        
        r = tj.Trajectory()
        x1 = np.random.rand(1000)
        x2 = np.random.rand(1000)
        x3 = np.random.rand(1000)
        x1x2 = np.array([x1, x2, x3]).transpose()
        r.gyration_radius = r.gyration_radius_(x1x2)['gyration tensor']
        r.eigenvalues, r.eigenvectors = r.gyration_radius_(x1x2)['eigenvalues'], r.gyration_radius_(x1x2)['eigenvectors']
        r.kurtosis = r.kurtosis_(x1x2, r.eigenvectors[:,0])
        self.assertGreaterEqual(r.kurtosis, 2.0)
        self.assertLessEqual(r.kurtosis, 3.0)


    def test_green_kubo(self):
        """
            testing the Green-Kubo diffusivity function
        """

        r = tj.Trajectory()
        velocity = np.ones(shape=(100,1))
        t = np.arange(0,100)
        velocity_autocorrelation = np.ones(shape=(100,1))
        self.assertAlmostEqual(r.green_kubo_(velocity, t, velocity_autocorrelation), 100.0, places=1)

    def test_velocity_description(self):
        '''
            testing the velocity description function
        '''
        r = tj.Trajectory()
        velocity = np.random.normal(size = (10000,1))
        velocity_description = r.velocity_description_(velocity)
        self.assertAlmostEqual(velocity_description['mean'][0], 0.0, places=1)
        self.assertAlmostEqual(velocity_description['median'][0], 0.0, places=1)
        self.assertAlmostEqual(velocity_description['standard_deviation'][0], 1.0, places=1)
        self.assertAlmostEqual(velocity_description['variance'][0], 1.0, places=1)
        self.assertAlmostEqual(velocity_description['range'][0], 7.5, places=1)
        self.assertAlmostEqual(velocity_description['kurtosis'][0], 0.0, places=1)
        self.assertAlmostEqual(velocity_description['skewness'][0], 0.0, places=1)

    def test_frequency_spectrum(self):
        '''
            testing the frequency spectrum function
        '''
        r = tj.Trajectory()
        t = np.linspace(0,1.0,10000).reshape(-1,1)
        x = 2*np.sin(2*np.pi*100*t).reshape(-1,1)
        frequency_spectrum = r.frequency_spectrum_(x,t)
        self.assertAlmostEqual(frequency_spectrum['dominant amplitude'][0], 2, places=1)
        self.assertAlmostEqual(frequency_spectrum['dominant frequency'][0], 100, places=1)

if __name__ == '__main__':
    unittest.main()
