import json

def convert_to_coco(input_json, output_json, images_dir):
    """
    Конвертирует JSON в COCO-формат.
    """
    with open(input_json, 'r') as f:
        data = json.load(f)

    CLASSES = ['Atelectasis', 'Calcification', 'Consolidation', 'Effusion',
               'Emphysema', 'Fibrosis', 'Fracture', 'Mass', 'Nodule', 'Pneumothorax']
    class_to_id = {cls: idx + 1 for idx, cls in enumerate(CLASSES)}  # COCO ID начинаются с 1

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": idx + 1, "name": cls} for idx, cls in enumerate(CLASSES)]
    }

    ann_id = 0
    for img_id, item in enumerate(data):
        coco_data["images"].append({
            "id": img_id + 1,
            "file_name": item['file_name'],
            "width": 1024,
            "height": 1024
        })

        # Аннотации
        for box, sym in zip(item['boxes'], item['syms']):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id + 1,
                "category_id": class_to_id[sym],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"✅ Конвертировано в COCO-формат: {output_json}")


if __name__ == "__main__":
    convert_to_coco(
        input_json="data/raw/ChestX-Det10-Dataset-master/test/test.json",
        output_json="data/raw/ChestX-Det10-Dataset-master/test/coco_test.json",
        images_dir="data/processed/test/images"
    )