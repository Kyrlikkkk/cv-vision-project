from ultralytics import YOLO

def load_yolov5(model_path='yolov5su.pt'):
    model = YOLO(model_path)
    return model