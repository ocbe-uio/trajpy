import numpy as np
import trajpy.trajpy as tj
import unittest


class TestGeneral(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.RandomState(0)

    def test_init(self):
        """
            test if the trajectory initialization is working properly
            initializing the trajectory with an array, a tuple or from file must
            give the same result
        """
        traj = np.zeros((370, 4))
        init_from_file = tj.Trajectory('../data/samples/sample.csv',
                                       skip_header=1, delimiter=',')
        traj[:, 0] = np.copy(init_from_file._t)
        traj[:, 1:] = np.copy(init_from_file._r)
        t = np.copy(traj[:, 0])
        x, y, z = np.copy(traj[:, 1]), np.copy(traj[:, 2]), np.copy(traj[:, 3])
        init_from_array = tj.Trajectory(traj)
        init_from_tuple = tj.Trajectory((t, x, y, z))
        self.assertEqual(init_from_array._r.all(), init_from_file._r.all())
        self.assertEqual(init_from_tuple._r.all(), init_from_file._r.all())


if __name__ == '__main__':
    unittest.main()
