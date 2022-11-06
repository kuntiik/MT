__all__ = [
    "ObjectDetectionRecord",
]

from mt.core import BaseRecord, FilepathRecordComponent, InstancesLabelsRecordComponent, BBoxesRecordComponent


def ObjectDetectionRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )
