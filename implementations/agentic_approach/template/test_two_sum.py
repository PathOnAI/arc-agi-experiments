import unittest
from two_sum import two_sum

class TestTwoSum(unittest.TestCase):
    def test_example_case(self):
        self.assertEqual(two_sum([2, 7, 11, 15], 9), [0, 1])

    def test_another_case(self):
        self.assertEqual(two_sum([3, 2, 4], 6), [1, 2])

    def test_large_numbers(self):
        self.assertEqual(two_sum([500000000, 1000000000], 1500000000), [0, 1])

    def test_negative_numbers(self):
        self.assertEqual(two_sum([-3, 4, 3, 90], 0), [0, 2])

    def test_same_numbers(self):
        self.assertEqual(two_sum([3, 3], 6), [0, 1])

if __name__ == "__main__":
    unittest.main()