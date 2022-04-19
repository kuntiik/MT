__all__ = ["Dataset"]

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL.ImageTransform import Transform
from PIL import Image

from src.core import ClassMap, BaseRecord, ImageRecordComponent, ClassMapRecordComponent, tasks, FilepathRecordComponent, SizeRecordComponent


class Dataset:
    """Container for a list of records and transforms.

    Steps each time an item is requested (normally via directly indexing the `Dataset`):
        * Grab a record from the internal list of records.
        * Prepare the record (open the image, open the mask, add metadata).
        * Apply transforms to the record.

    # Arguments
        records: A list of records.
        tfm: Transforms to be applied to each item.
    """

    def __init__(
            self,
            records: List[dict],
            tfm: Optional[Transform] = None,
    ):
        self.records = records
        self.tfm = tfm
        # if self.tfm is not None:
        #     self.tfm.setup(records[0].components_cls)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i].load()
        if self.tfm is not None:
            record = self.tfm(record)
        else:
            # HACK FIXME
            record.set_img(np.array(record.img))
        return record

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.records)} items>"

    @classmethod
    def from_images(
            cls,
            images: Sequence[np.array],
            tfm: Transform = None,
            class_map: Optional[ClassMap] = None,
    ):
        """Creates a `Dataset` from a list of images.

        # Arguments
            images: `Sequence` of images in memory (numpy arrays).
            tfm: Transforms to be applied to each item.
        """
        records = []
        for i, image in enumerate(images):
            record = BaseRecord((ImageRecordComponent(),))
            record.set_record_id(i)
            record.set_img(image)
            records.append(record)

            record.add_component(ClassMapRecordComponent(task=tasks.detection))
            if class_map is not None:
                record.detection.set_class_map(class_map)

        return cls(records=records, tfm=tfm)

    @classmethod
    def from_image_folder(cls, filepath: str, tfm: Transform = None, class_map: Optional[ClassMap] = None):
        records = []
        folder_path = Path(filepath)
        for image in folder_path.iterdir():
            if image.suffix not in ['.jpg', '.png', '.jpeg']:
                continue
            img = Image.open(image)
            width, height = img.size
            record = BaseRecord((FilepathRecordComponent(), SizeRecordComponent()))
            record.set_filepath(image)
            record.set_img_size(original=True)
            record.set_record_id(image)
            record.add_component(ClassMapRecordComponent(task=tasks.detection))
            if class_map is not None:
                record.detection.set_class_map(class_map)
            records.append(record)
        return cls(records=records, tfm=tfm)


