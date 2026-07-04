import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def load_faster_rcnn(num_classes=11):
    """
    Загружает Faster R-CNN с бекбоном ResNet-50.
    num_classes = количество классов + 1 (фон) = 11
    """
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Меняем голову на наши 10 классов (+ фон)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    return model