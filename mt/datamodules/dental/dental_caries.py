import pathlib
from pathlib import Path
from typing import Hashable, Any, List, Optional

from mt.core import ClassMap, BBox
from mt.core.record_defaults import ObjectDetectionRecord
from mt.data.dataset import Dataset
from mt.data.parser import Parser
from mt.data.random_splitter import RandomSplitter
from mt.utils import ImgSize
import mt.models
from mt.models import *

from pycocotools.coco import COCO
import pytorch_lightning as pl


class DentalCariesParser(Parser):
    def __init__(self, data_root, annotation_file):
        template_record = ObjectDetectionRecord()
        super().__init__(template_record=template_record)
        # if data_root is not Path:
        if not isinstance(data_root, pathlib.Path):
            data_root = Path(data_root)
        self.coco = COCO(data_root / annotation_file)
        self.data_root = Path(data_root)
        self.class_map = ClassMap(["decay"])
        self.prepare_coco()

    def prepare_coco(self):
        img_ids = list(sorted(self.coco.imgs.keys()))
        self.data = [self._load_image_coco(img_id) for img_id in img_ids]

    def _load_image_coco(self, id):
        img = self.coco.loadImgs(id)[0]
        targets = self.coco.loadAnns(self.coco.getAnnIds(id))
        return img, targets

    def __iter__(self) -> Any:
        for o in self.data:
            yield o

    def __len__(self) -> int:
        return len(self.data)

    def record_id(self, o) -> Hashable:
        img, target = o
        return img['file_name']

    def parse_fields(self, o, record, is_new):
        img, targets = o
        record.set_filepath(self.data_root / "images" / img['file_name'])
        record.set_img_size(ImgSize(width=img['width'], height=img['height']), original=True)
        record.detection.set_class_map(self.class_map)

        for target in targets:
            record.detection.add_bboxes([BBox.from_xywh(*target['bbox'])])
            record.detection.add_labels(["decay"])


class DentalCaries(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        model_type,
        ann_file: str = "annotations.json",
        batch_size: int = 4,
        num_workers: int = 4,
        seed=42,
        train_transforms=None,
        val_transforms=None,
        train_val_test_split: List[int] = [0.8, 0.1, 0.1],
        *args,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_type", "transforms"])
        self.batch_size = batch_size
        mod = mt.models
        for module in model_type.split("."):
            mod = getattr(mod, module)
        self.model_type = mod
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None):
        template_record = ObjectDetectionRecord()
        parser = DentalCariesParser(self.hparams.data_root, self.hparams.ann_file)
        train_record, valid_record, test_record = parser.parse(
            data_splitter=RandomSplitter(
                self.hparams.train_val_test_split, seed=self.hparams.seed
            )
        )
        self.train_ds = Dataset(train_record, self.hparams.train_transforms)
        self.valid_ds = Dataset(valid_record, self.hparams.val_transforms)
        self.test_ds = Dataset(test_record, self.hparams.val_transforms)

    def train_dataloader(self):
        return self.model_type.train_dl(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True
        )

    def val_dataloader(self: Optional[str] = None):
        return self.model_type.valid_dl(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self: Optional[str] = None):
        return self.model_type.valid_dl(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self, stage="test"):
        if stage == "test":
            ds = self.test_ds
        elif stage == "val":
            ds = self.valid_ds
        # elif stage ==  'train':
        else:
            ds = self.train_ds
        return self.model_type.infer_dl(
            ds,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            shuffle=False
        )
