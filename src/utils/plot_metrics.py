import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

PROJECT_ROOT = Path(__file__).parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "results/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

model_dirs = [
    PROJECT_ROOT / "results/yolo",
    PROJECT_ROOT / "results/yolov5",
    PROJECT_ROOT / "results/yolov5m",
    PROJECT_ROOT / "results/faster_rcnn",
    PROJECT_ROOT / "results/ssd",
    PROJECT_ROOT / "results/retinanet"
]

metrics_list = []
for dir_path in model_dirs:
    json_path = Path(dir_path) / "metrics.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
            metrics_list.append(data)
            print(f"Загружено: {data['model']}")
    else:
        print(f"Файл не найден: {json_path}")

# Преобразуем в DataFrame
df = pd.DataFrame(metrics_list)
print("\nСводная таблица:")
print(df.to_string())

colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6', '#7f8c8d']

#График 1: Сравнение mAP@0.5
plt.figure(figsize=(12, 6))
bars = plt.bar(df['model'], df['map50'], color=colors[:len(df)], edgecolor='black', linewidth=1.2)
plt.ylim(0, 0.4)
plt.ylabel('mAP@0.5', fontsize=14)
plt.xlabel('Модель', fontsize=14)
plt.title('Сравнение моделей по точности (mAP@0.5)', fontsize=16)
plt.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, df['map50']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "comparison_map50.png", dpi=150)
plt.close()

# График 2: Время обучения vs mAP
plt.figure(figsize=(12, 6))
plt.scatter(df['time_hours'], df['map50'], s=200, c=colors[:len(df)], edgecolors='black', linewidth=1.5)
for i, row in df.iterrows():
    plt.annotate(row['model'], (row['time_hours'], row['map50']),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=11, fontweight='bold')
plt.xlabel('Время обучения (часы)', fontsize=14)
plt.ylabel('mAP@0.5', fontsize=14)
plt.title('Соотношение времени обучения и точности', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "comparison_time_vs_accuracy.png", dpi=150)
plt.close()

# График 3: Precision, Recall, F1 для YOLO
yolo_df = df[df['model'].str.contains('YOLO')]
if not yolo_df.empty:
    x = np.arange(len(yolo_df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, yolo_df['precision'], width, label='Precision', color='#3498db')
    plt.bar(x, yolo_df['recall'], width, label='Recall', color='#2ecc71')
    plt.bar(x + width, yolo_df['f1'], width, label='F1-score', color='#f39c12')

    plt.xticks(x, yolo_df['model'])
    plt.ylabel('Значение', fontsize=14)
    plt.title('Сравнение Precision, Recall, F1 для моделей YOLO', fontsize=16)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    for i, row in yolo_df.iterrows():
        idx = i
        plt.text(idx - width, row['precision'] + 0.02, f'{row["precision"]:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(idx, row['recall'] + 0.02, f'{row["recall"]:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(idx + width, row['f1'] + 0.02, f'{row["f1"]:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison_precision_recall_f1.png", dpi=150)
    plt.close()
