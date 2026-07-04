import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ssd import load_ssd
from src.dataset.dataset import get_dataloader

EPOCHS = 50
BATCH_SIZE = 4
LR = 0.0005
MOMENTUM = 0.937

TRAIN_IMAGES = Path("data/processed/train/images")
TRAIN_LABELS = Path("data/processed/train/labels")
RESULTS_DIR = Path("results/ssd")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

model = load_ssd(num_classes=11)
model.to(0)

train_loader = get_dataloader(TRAIN_IMAGES, TRAIN_LABELS, batch_size=BATCH_SIZE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

print(f"Переобучение SSD на {EPOCHS} эпох...")

history = {'loss': []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batch_count = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        has_objects = any(len(t['boxes']) > 0 for t in targets)
        if not has_objects:
            continue

        images = [img.to(0) for img in images]
        targets = [{k: v.to(0) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        batch_count += 1

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {losses.item():.4f}")

    if batch_count > 0:
        avg_loss = total_loss / batch_count
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")
    else:
        print(f"Epoch {epoch + 1}/{EPOCHS}, нет батчей с объектами!")

torch.save(model.state_dict(), RESULTS_DIR / "model_retrained.pth")

with open(RESULTS_DIR / "history_retrained.json", "w") as f:
    json.dump(history, f, indent=4)

# График потерь
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(history['loss']) + 1), history['loss'], 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SSD Retrained Loss')
plt.grid(True, alpha=0.3)
plt.savefig(RESULTS_DIR / "loss_plot_retrained.png", dpi=150)
plt.close()
