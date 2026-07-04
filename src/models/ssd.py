from torchvision.models.detection import ssd300_vgg16


def load_ssd(num_classes=11):
    """
    Загружает SSD300 с бекбоном VGG16.
    num_classes = количество классов + 1 (фон) = 11
    """
    model = ssd300_vgg16(weights='DEFAULT')
    model.head.classification_head.num_classes = num_classes

    return model