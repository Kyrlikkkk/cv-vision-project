from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.yolo import load_yolo

def run_training():
    model = load_yolo('yolov8n.pt')

    # Путь к YAML-файлу с данными
    data_yaml_path = Path("configs/chestxdet.yaml")

    results = model.train(
        data=str(data_yaml_path),
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        patience=10,  # остановка, если нет улучшений 10 эпох
        save=True,
        project="results/yolo",
        name="chestxdet_run1",
        exist_ok=True,
        verbose=True,
    )
if __name__ == '__main__':
    run_training()