import json
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results/faster_rcnn")

with open(RESULTS_DIR / "history.json", "r") as f:
    history = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(history['loss'])+1), history['loss'], 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Faster R-CNN Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig(RESULTS_DIR / "loss_plot.png", dpi=150)
plt.show()
print(f"График сохранён в {RESULTS_DIR / 'loss_plot.png'}")