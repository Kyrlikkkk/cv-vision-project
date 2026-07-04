import torch
from pathlib import Path
import sys
import json
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.retinanet import load_retinanet

TEST_JSON = Path("data/raw/ChestX-Det10-Dataset-master/test/coco_test.json")
TEST_IMAGES = Path("data/processed/test/images")
RESULTS_DIR = Path("results/retinanet_official")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Загрузка модели
model = load_retinanet(num_classes=11)
model.load_state_dict(torch.load("results/retinanet/model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Модель RetinaNet загружена")


#Датасет в формате COCO
class ChestXDetDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, json_file):
        self.img_dir = Path(img_dir)
        self.coco = COCO(json_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(self.img_dir / path).convert("RGB")
        return F.to_tensor(img), img_id, anns

    def __len__(self):
        return len(self.ids)


dataset = ChestXDetDataset(TEST_IMAGES, TEST_JSON)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

results = []
with torch.no_grad():
    for images, img_ids, targets in dataloader:
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)
        for output, img_id in zip(outputs, img_ids):
            for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                if score > 0.01:  # порог уверенности
                    x1, y1, x2, y2 = box.cpu().numpy()
                    results.append({
                        'image_id': int(img_id),
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        'score': float(score)
                    })

pred_path = RESULTS_DIR / "predictions.json"
with open(pred_path, 'w') as f:
    json.dump(results, f)

print(f"Сохранено {len(results)} предсказаний в {pred_path}")
#Метрики
try:
    coco_gt = COCO(TEST_JSON)
    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\nРезультаты RetinaNet (COCOeval):")
    print(f"  mAP@0.5: {coco_eval.stats[1]:.4f}")
    print(f"  mAP@0.5:0.95: {coco_eval.stats[0]:.4f}")

    metrics = {
        'model': 'RetinaNet',
        'map_50': coco_eval.stats[1],
        'map': coco_eval.stats[0]
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Метрики сохранены в {RESULTS_DIR / 'metrics.json'}")

except Exception as e:
    print(f"Ошибка при вычислении метрик: {e}")