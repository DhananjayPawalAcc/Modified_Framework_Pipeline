from model import YOLOv8DetectionModel, YOLOv8SegmentationModel, YOLOv8ClassificationModel

class ModelInitializer:
    def __init__(self, model_type='yolov8_detection', model_path=None):
        self.model_type = model_type
        # self.model_path = model_path
        self.model_path = model_path if model_path else 'yolov8n.pt'
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == 'yolov8_detection':
            return YOLOv8DetectionModel(self.model_path)
        elif self.model_type == 'yolov8_segmentation':
            return YOLOv8SegmentationModel(self.model_path)
        elif self.model_type == 'yolov8_classification':
            return YOLOv8ClassificationModel(self.model_path)
        else:
            raise ValueError("Unsupported model type.")

    def get_model(self):
        return self.model
