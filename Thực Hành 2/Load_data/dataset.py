from pathlib import Path
import struct
import numpy as np


class MNISTDataset:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent.parent / "data"
        self.x_train = self.y_train = self.x_test = self.y_test = None

    @staticmethod
    def load_images(path):
        with path.open("rb") as file:
            header = file.read(16)
            magic, count, rows, cols = struct.unpack(">IIII", header)
            images = np.frombuffer(file.read(), dtype=np.uint8)
        return images.reshape(count, rows * cols).astype(np.float32) / 255.0

    @staticmethod
    def load_labels(path):
        with path.open("rb") as file:
            header = file.read(8)
            magic, count = struct.unpack(">II", header)
            labels = np.frombuffer(file.read(), dtype=np.uint8).astype(np.int64)
        return labels

    def load(self):
        self.x_train = self.load_images(self.data_dir / "train-images-idx3-ubyte")
        self.y_train = self.load_labels(self.data_dir / "train-labels-idx1-ubyte")
        self.x_test = self.load_images(self.data_dir / "t10k-images-idx3-ubyte")
        self.y_test = self.load_labels(self.data_dir / "t10k-labels-idx1-ubyte")
        return self.x_train, self.y_train, self.x_test, self.y_test
