import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.faster_rcnn import load_faster_rcnn
from src.dataset.dataset import get_dataloader

EPOCHS = 30
BATCH_SIZE = 4
LR = 0.01
MOMENTUM = 0.937

TRAIN_IMAGES = Path("data/processed/train/images")
TRAIN_LABELS = Path("data/processed/train/labels")

model = load_faster_rcnn(num_classes=11)
model.to(0)

#Данные
train_loader = get_dataloader(TRAIN_IMAGES, TRAIN_LABELS, batch_size=BATCH_SIZE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

print(f"Обучение Faster R-CNN на {EPOCHS} эпох...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(0) for img in images]
        targets = [{k: v.to(0) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

Path("results/faster_rcnn").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), "results/faster_rcnn/model.pth")
