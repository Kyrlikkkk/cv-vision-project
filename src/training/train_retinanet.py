import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.retinanet import load_retinanet
from src.dataset.dataset import get_dataloader

EPOCHS = 30
BATCH_SIZE = 4
LR = 0.001
MOMENTUM = 0.937

TRAIN_IMAGES = Path("data/processed/train/images")
TRAIN_LABELS = Path("data/processed/train/labels")
RESULTS_DIR = Path("results/retinanet")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

model = load_retinanet(num_classes=11)
model.to(0)

#данные
train_loader = get_dataloader(TRAIN_IMAGES, TRAIN_LABELS, batch_size=BATCH_SIZE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

print(f"Обучение RetinaNet на {EPOCHS} эпох...")

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
        print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")
    else:
        print(f"Epoch {epoch + 1}/{EPOCHS}, нет батчей с объектами!")

torch.save(model.state_dict(), RESULTS_DIR / "model.pth")
