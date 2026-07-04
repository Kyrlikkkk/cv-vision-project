from ultralytics import YOLO


def load_yolo(model_path='yolov8n.pt'):
    model = YOLO(model_path)
    return model