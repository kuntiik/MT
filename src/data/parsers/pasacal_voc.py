from pathlib import Path
from typing import Optional, Hashable, List, Union

from src.core import ClassMap, BaseRecord, FilepathRecordComponent, InstancesLabelsRecordComponent, \
    BBoxesRecordComponent, BBox
from src.core.class_map import IDMap
from src.data.parser import Parser

import xml.etree.ElementTree as ET

from src.utils import ImgSize


class VOCBBoxParser(Parser):
    def __init__(
            self,
            annotations_dir: Union[str, Path],
            images_dir: Union[str, Path],
            class_map: Optional[ClassMap] = None,
            idmap: Optional[IDMap] = None,
    ):
        super().__init__(template_record=self.template_record(), idmap=idmap)
        self.class_map = class_map or ClassMap().unlock()
        self.images_dir = Path(images_dir)

        self.annotations_dir = Path(annotations_dir)
        self.annotation_files = sorted(list(self.annotations_dir.iterdir()))

    def __len__(self):
        return len(self.annotation_files)

    def __iter__(self):
        yield from self.annotation_files

    def template_record(self) -> BaseRecord:
        return BaseRecord(
            (
                FilepathRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
            )
        )

    def record_id(self, o) -> Hashable:
        return str(Path(self._filename).stem)

    def prepare(self, o):
        tree = ET.parse(str(o))
        self._root = tree.getroot()
        self._filename = self._root.find("filename").text
        self._size = self._root.find("size")

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(self.filepath(o))
            record.set_img_size(self.img_size(o), original=True)

        record.detection.set_class_map(self.class_map)
        record.detection.add_labels(self.labels(o))
        record.detection.add_bboxes(self.bboxes(o))

    def filepath(self, o) -> Union[str, Path]:
        return self.images_dir / self._filename

    def img_size(self, o) -> ImgSize:
        width = int(self._size.find("width").text)
        height = int(self._size.find("height").text)
        return ImgSize(width=width, height=height)

    def labels(self, o) -> List[Hashable]:
        labels = []
        for object in self._root.iter("object"):
            label = object.find("name").text
            labels.append(label)

        return labels

    def bboxes(self, o) -> List[BBox]:
        def to_int(x):
            return int(float(x))

        bboxes = []
        for object in self._root.iter("object"):
            xml_bbox = object.find("bndbox")
            xmin = to_int(xml_bbox.find("xmin").text)
            ymin = to_int(xml_bbox.find("ymin").text)
            xmax = to_int(xml_bbox.find("xmax").text)
            ymax = to_int(xml_bbox.find("ymax").text)

            bbox = BBox.from_xyxy(xmin, ymin, xmax, ymax)
            bboxes.append(bbox)

        return bboxes
