__all__ = [
    "ObjectDetectionRecord",
]

from src.core import BaseRecord, FilepathRecordComponent, InstancesLabelsRecordComponent, BBoxesRecordComponent


def ObjectDetectionRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )
