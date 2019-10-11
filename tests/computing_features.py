import sys
sys.path.insert(0, '../trajpy')
import numpy as np
import trajpy as tj
import unittest


class TestFeatures(unittest.TestCase):

    def test_efficiency(self):
        """
            test if the efficiency is being computed correctly for two extreme cases
            1) linear trajectory and 2) random steps.
        """

        x1 = np.linspace(0, 100, 100)
        x2 = np.random.rand(100)

        r = tj.Trajectory()

        self.assertAlmostEqual(r.efficiency_(x1), 1.0)
        self.assertAlmostEqual(r.efficiency_(x2), 0.0)


if __name__ == '__main__':
    unittest.main()
