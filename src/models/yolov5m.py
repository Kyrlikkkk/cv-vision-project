from ultralytics import YOLO

def load_yolov5m(model_path='yolov5mu.pt'):
    model = YOLO(model_path)
    return model