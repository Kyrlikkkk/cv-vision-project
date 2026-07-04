from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.yolov5m import load_yolov5m

def run_training():
    model = load_yolov5m('yolov5mu.pt')
    data_yaml_path = Path("configs/chestxdet.yaml")

    model.train(
        data=str(data_yaml_path),
        epochs=30,
        imgsz=640,
        batch=6,
        device=0,
        workers=4,
        patience=10,
        save=True,
        project="results/yolov5m",
        name="chestxdet_run1",
        exist_ok=True,
        verbose=True,
        val=False
    )


if __name__ == '__main__':
    run_training()