import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path


class ChestXDetDataset(Dataset):
    def __init__(self, images_dir, labels_dir, target_size=None, normalize=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_paths = list(self.images_dir.glob("*.png"))
        self.target_size = target_size  # (width, height) или None
        self.normalize = normalize

        # Стандартная нормализация ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Если задан target_size — изменяем размер
        if self.target_size is not None:
            img = cv2.resize(img, self.target_size)
            new_h, new_w = img.shape[:2]
            scale_x = new_w / w
            scale_y = new_h / h
        else:
            new_h, new_w = h, w
            scale_x, scale_y = 1.0, 1.0

        # Читаем разметку
        label_path = self.labels_dir / (img_path.stem + ".txt")
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        xc = float(parts[1]) * w
                        yc = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h

                        x1 = (xc - bw / 2) * scale_x
                        y1 = (yc - bh / 2) * scale_y
                        x2 = (xc + bw / 2) * scale_x
                        y2 = (yc + bh / 2) * scale_y

                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Преобразуем в тензор
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Нормализация
        if self.normalize:
            img = (img - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor(
                [0.0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return img, target


def get_dataloader(images_dir, labels_dir, batch_size=4, shuffle=True, target_size=None, normalize=True):
    dataset = ChestXDetDataset(images_dir, labels_dir, target_size, normalize)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: tuple(zip(*x))
    )