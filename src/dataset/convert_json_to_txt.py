import json
from pathlib import Path

RAW_TRAIN_JSON = Path("data/raw/ChestX-Det10-Dataset-master/train/train.json")
RAW_TEST_JSON = Path("data/raw/ChestX-Det10-Dataset-master/test/test.json")

LABELS_TRAIN_DIR = Path("data/processed/train/labels")
LABELS_TEST_DIR = Path("data/processed/test/labels")

# 1. Загружаем JSON
with open(RAW_TRAIN_JSON, "r") as f:
    train_data = json.load(f)

with open(RAW_TEST_JSON, "r") as f:
    test_data = json.load(f)

# 2. Собираем все уникальные классы и создаём словарь {класс: id}
all_classes = set()
for item in train_data + test_data:
    for cls in item["syms"]:
        all_classes.add(cls)

class_to_id = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
print(f"Найдено классов: {len(class_to_id)}")
print("Список классов:", class_to_id)

# 3. Функция для конвертации одного элемента
def convert_item(item, img_width=1024, img_height=1024):
    """
    Принимает один элемент из JSON.
    Возвращает список строк для .txt файла в формате YOLO.
    """
    lines = []
    for box, sym in zip(item["boxes"], item["syms"]):
        x1, y1, x2, y2 = box

        # Нормализуем координаты (приводим к диапазону 0..1)
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        class_id = class_to_id[sym]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines

# 4. Конвертируем train
for item in train_data:
    filename = item["file_name"]
    txt_filename = filename.replace(".png", ".txt")
    txt_path = LABELS_TRAIN_DIR / txt_filename

    lines = convert_item(item)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

print(f"Создано {len(train_data)} файлов в {LABELS_TRAIN_DIR}")

# 5. Конвертируем test
for item in test_data:
    filename = item["file_name"]
    txt_filename = filename.replace(".png", ".txt")
    txt_path = LABELS_TEST_DIR / txt_filename

    lines = convert_item(item)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

print(f"Создано {len(test_data)} файлов в {LABELS_TEST_DIR}")

