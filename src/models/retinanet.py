from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def load_retinanet(num_classes=11):
    """
    Загружает RetinaNet с бекбоном ResNet-50.
    num_classes = количество классов + 1 (фон) = 11
    """
    model = retinanet_resnet50_fpn(weights='DEFAULT')

    old_head = model.head.classification_head
    in_channels = old_head.conv[0].out_channels
    num_anchors = old_head.num_anchors

    # Создаём новую голову
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    return model