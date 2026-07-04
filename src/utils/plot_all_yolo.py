import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

yolo_models = [
    {"name": "YOLOv8n", "path": "runs/detect/results/yolo/chestxdet_run1/results.csv"},
    {"name": "YOLOv5s", "path": "runs/detect/results/yolov5/chestxdet_run1/results.csv"},
    {"name": "YOLOv5m", "path": "runs/detect/results/yolov5m/chestxdet_run1/results.csv"}
]

for model in yolo_models:
    try:
        df = pd.read_csv(model["path"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # График 1: Loss (box_loss, cls_loss, dfl_loss)
        if 'train/box_loss' in df.columns:
            axes[0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
            axes[0].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', linewidth=2)
            axes[0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model["name"]} - Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # График 2: Метрики (mAP50, Precision, Recall)
        if 'metrics/mAP_0.5' in df.columns:
            axes[1].plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5', linewidth=2)
        if 'metrics/precision' in df.columns:
            axes[1].plot(df['epoch'], df['metrics/precision'], label='Precision', linewidth=2)
        if 'metrics/recall' in df.columns:
            axes[1].plot(df['epoch'], df['metrics/recall'], label='Recall', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title(f'{model["name"]} - Quality Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{model['name']}_individual.png", dpi=150)
        plt.close()

    except Exception as e:
        print(f"Ошибка для {model['name']}: {e}")

