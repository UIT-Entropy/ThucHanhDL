import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, images, labels, image_size=28):
        self.images = torch.from_numpy(images).reshape(-1, 1, 28, 28)
        self.labels = torch.from_numpy(labels)
        self.image_size = image_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.image_size != 28:
            image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
        return image, self.labels[index]
