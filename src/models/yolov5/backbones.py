__all__ = [
    "small",
    "small_p6",
    "medium",
    "medium_p6",
    "large",
    "large_p6",
    "extra_large",
    "extra_large_p6",
    "YoloV5BackboneConfig",
]


class YoloV5BackboneConfig:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pretrained: bool

    def __call__(self, pretrained: bool = True) -> "YoloV5BackboneConfig":
        """Completes the configuration of the backbone

        # Arguments
            pretrained: If True, use a pretrained backbone (on COCO).
        """
        self.pretrained = pretrained
        return self


small = YoloV5BackboneConfig(model_name="yolov5s")
small_p6 = YoloV5BackboneConfig(model_name="yolov5s6")

medium = YoloV5BackboneConfig(model_name="yolov5m")
medium_p6 = YoloV5BackboneConfig(model_name="yolov5m6")

large = YoloV5BackboneConfig(model_name="yolov5l")
large_p6 = YoloV5BackboneConfig(model_name="yolov5l6")

extra_large = YoloV5BackboneConfig(model_name="yolov5x")
extra_large_p6 = YoloV5BackboneConfig(model_name="yolov5x6")
