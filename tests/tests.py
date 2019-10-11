sys.path.insert(0, '../trajpy')
import trajpy as tj
import unittest



class TestFeatures(unittest.TestCase):

    def test_object_feature(self):
      r = tj.Features()
      self.assertEqual(r, tj.Features())


if __name__ == '__main__':
    unittest.main()
