import numpy as np
import trajpy.trajpy as tj
import unittest


class TestFeatures(unittest.TestCase):
 
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_efficiency(self):
        """
            test if the efficiency is being computed correctly for two extreme cases
            1) linear trajectory and 2) random steps.
        """

        x1 = np.linspace(0, 100, 100)
        x2 = np.random.rand(100)

        r = tj.Trajectory()

        self.assertAlmostEqual(r.efficiency_(x1), 1.0, places=2)
        self.assertAlmostEqual(r.efficiency_(x2), 0.0, places=2)

    def test_fractal_dimension(self):
        """
            test if the fractal dimension is being computed correctly
        """

        x1 = np.linspace(0, 100, 100)
        x2 = np.random.rand(100)

        r = tj.Trajectory()

        self.assertAlmostEqual(r.fractal_dimension_(x1), 1.0, places=2)
        self.assertGreaterEqual(r.fractal_dimension_(x2), 3.0)


if __name__ == '__main__':
    unittest.main()
