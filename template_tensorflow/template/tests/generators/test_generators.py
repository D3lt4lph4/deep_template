import unittest

from keras.datasets import mnist

from template.generators import Generator

import numpy as np


class TestVariables(unittest.TestCase):

    def setUp(self):
        # We create a generator
        data, val = mnist.load_data()
        self.generator_1 = Generator(data)
        self.generator_2 = Generator(val, batch_size=64, image_shape=(28, 28), shuffle=False, num_classes=10)

    def test_batch_size(self):
        self.assertEqual(self.generator_1.batch_size, 32)
        self.assertEqual(self.generator_2.batch_size, 64)

    def test_number_of_samples(self):
        self.assertEqual(self.generator_1.number_of_data_samples, 60000)
        self.assertEqual(self.generator_2.number_of_data_samples, 10000)
    
    def test_shuffle(self):
        self.assertTrue(self.generator_1.shuffle)
        self.assertFalse(self.generator_2.shuffle)

    def test_shuffle_batches(self):
        _, y_1 = self.generator_1.__getitem__(1)
        self.generator_1.shuffle_batches()
        _, y_2 = self.generator_1.__getitem__(1)

        equal = True
        for i, y in enumerate(y_1):
            equal &= np.array_equal(y, y_2[i])

        self.assertFalse(equal)
    
    def test_len(self):
        self.assertEqual(len(self.generator_1), 1875)
        self.assertEqual(len(self.generator_2), 156)

    def test_get_item(self):
        X, y = self.generator_2.__getitem__(0)

        print(y)

        self.assertListEqual([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], y[0].tolist())

    def test_get_batch_data(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
