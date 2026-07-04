from ultralytics import YOLO

def run_validation():
    # Загружаем модель
    model = YOLO("runs/detect/results/yolov5m/chestxdet_run1/weights/best.pt")

    # Запускаем валидацию на тестовой выборке
    results = model.val(
        data="configs/chestxdet.yaml",
        split="test",       # используем тестовую выборку
        conf=0.001,
        iou=0.6,
        batch=1,
        imgsz=640,
        device=0,
        plots=True,
        save_json=True,
        save_txt=True,
    )

    # Выводим метрики
    print("\nРезультаты валидации YOLOv5m:")
    print(f"  mAP@0.5: {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")

if __name__ == '__main__':
    run_validation()