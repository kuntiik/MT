import unittest

from src.lightning_datamodules.data.random_splitter import SingleSplitSplitter, RandomSplitter
from src.lightning_datamodules.data.record_collection import RecordCollection
from src.lightning_datamodules.dental_rtg.dental_caries import DentalCariesParser, DentalCaries
from pathlib import Path
import src.lightning_modules.models.efficientdet


class TestDentalCariesParser(unittest.TestCase):
    data_root = Path("data/dataset")
    ann_file = "annotations.json"
    parser = DentalCariesParser(data_root, ann_file)

    def test_parsing(self):
        records = self.parser.parse(SingleSplitSplitter())[0]
        self.assertEqual(len(records), 3)

    def test_spltted_parsing(self):
        train, val, test = self.parser.parse(RandomSplitter([1 / 3, 1 / 3, 1 / 3], seed=42))
        train2, val2, test2 = self.parser.parse(RandomSplitter([1 / 3, 1 / 3, 1 / 3], seed=42))
        self.assertEqual(len(train) + len(val) + len(test), 3)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)
        self.assertEqual(train[0].record_id, train2[0].record_id)
        self.assertEqual(val[0].record_id, val2[0].record_id)
        self.assertEqual(test[0].record_id, test2[0].record_id)

    def test_type(self):
        records = self.parser.parse(SingleSplitSplitter())[0]
        self.assertIs(type(records), RecordCollection)

    def test_bboxes(self):
        records = self.parser.parse(SingleSplitSplitter())[0]
        self.assertEqual(len(records.get_by_record_id('1.png').detection.bboxes), 4)
        self.assertEqual(len(records.get_by_record_id('10.png').detection.bboxes), 2)
        self.assertEqual(len(records.get_by_record_id('100.png').detection.bboxes), 0)

    def test_orig_img_size(self):
        record = self.parser.parse(SingleSplitSplitter())[0][0]
        self.assertEqual(record.img_size, record.original_img_size)


class TestDentalCaries(unittest.TestCase):
    # data_root = Path("data/dataset")
    data_root = "data/dataset"
    ann_file = "annotations.json"

    def test_loaders(self):
        dm = DentalCaries(self.data_root, 'efficientdet', self.ann_file, 1, 0, train_val_test_split=[1 / 3, 1 / 3, 1 / 3])
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        self.assertEqual(len(train_loader) + len(val_loader) + len(test_loader), 3)
