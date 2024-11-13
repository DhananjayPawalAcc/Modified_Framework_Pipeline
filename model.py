from ultralytics import YOLO

class BaseModel:
    """A base class for all models. Define the interface here."""
    def predict(self, frame):
        raise NotImplementedError("Predict method should be implemented by the specific model subclass!!!")

class YOLOv8DetectionModel(BaseModel):
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model(frame, task="detect")  # Detection task
        return results[0]

class YOLOv8SegmentationModel(BaseModel):
    def __init__(self, model_path='yolov8n-seg.pt'):
        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model(frame, task="segment")  # Segmentation task
        return results[0]

class YOLOv8ClassificationModel(BaseModel):
    def __init__(self, model_path='yolov11n-cls.pt'):
        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model(frame, task="classify")  # Classification task
        return results[0]
