import unittest

import datasets
from datasets import Dataset


class DatasetTestCase(unittest.TestCase):
    def test_load_from_dict(self):
        my_dict = {"a": [1, 2, 3]}
        dataset = Dataset.from_dict(my_dict)
        assert dataset["a"] == [1, 2, 3]

    def test_split_concatenate(self):
        ri = datasets.ReadInstruction("train") + datasets.ReadInstruction("test")
        train_test_ds = datasets.load_dataset("bookcorpus", split=ri)
        print(train_test_ds)
