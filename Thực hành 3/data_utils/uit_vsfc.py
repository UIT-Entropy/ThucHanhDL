import json
import string
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_fn(items):
    inputs = [item["input_ids"] for item in items]
    lengths = torch.tensor([len(ids) for ids in inputs], dtype=torch.long)
    labels = torch.stack([item["label"] for item in items])
    return {"input_ids": pad_sequence(inputs, batch_first=True, padding_value=0), "lengths": lengths, "label": labels}


class Vocab:
    def __init__(self, train_path, label_key="topic"):
        self.label_key = label_key
        self.pad = "<pad>"
        self.unk = "<unk>"
        data = json.loads(Path(train_path).read_text(encoding="utf-8"))
        words = set()
        labels = set()
        for item in data:
            words.update(self.preprocess_sentence(item["sentence"]).split())
            labels.add(item[label_key])
        self.w2i = {self.pad: 0, self.unk: 1}
        self.w2i.update({word: index for index, word in enumerate(sorted(words - {self.pad, self.unk}), start=2)})
        self.i2w = {index: word for word, index in self.w2i.items()}
        self.l2i = {label: index for index, label in enumerate(sorted(labels))}
        self.i2l = {index: label for label, index in self.l2i.items()}

    @property
    def n_labels(self):
        return len(self.l2i)

    @property
    def len(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return 0

    def preprocess_sentence(self, sentence):
        return sentence.lower().translate(str.maketrans("", "", string.punctuation))

    def encode_sentence(self, sentence):
        words = self.preprocess_sentence(sentence).split() or [self.unk]
        return torch.tensor([self.w2i.get(word, 1) for word in words], dtype=torch.long)

    def encode_label(self, label):
        return torch.tensor(self.l2i[label], dtype=torch.long)

    def decode_label(self, label_ids):
        return [self.i2l[index] for index in label_ids.tolist()]


class UIT_VSFC(Dataset):
    def __init__(self, path, vocab):
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return {"input_ids": self.vocab.encode_sentence(item["sentence"]), "label": self.vocab.encode_label(item[self.vocab.label_key])}
