from pathlib import Path


class VinaFoodDataset:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent.parent / "data" / "VinaFood21"
        if not (self.data_dir / "train").is_dir() and (self.data_dir / "VinaFood21").is_dir():
            self.data_dir = self.data_dir / "VinaFood21"

    def load(self, train_transform=None, test_transform=None):
        from torchvision import datasets, transforms

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if train_transform is None:
            train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        if test_transform is None:
            test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
        train_dataset = datasets.ImageFolder(self.data_dir / "train", transform=train_transform)
        test_dataset = datasets.ImageFolder(self.data_dir / "test", transform=test_transform)
        if train_dataset.class_to_idx != test_dataset.class_to_idx or len(train_dataset.classes) != 21:
            raise ValueError("Train và test phải có cùng 21 lớp VinaFood21")
        return train_dataset, test_dataset
