from torch import nn


class ResNet50(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        import timm

        self.network = timm.create_model("hf-hub:timm/resnet50.a1_in1k", pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.network(x)

    def transforms(self):
        from timm.data import resolve_model_data_config, create_transform

        config = resolve_model_data_config(self.network)
        return create_transform(**config, is_training=True), create_transform(**config, is_training=False)
