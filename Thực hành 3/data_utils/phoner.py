from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_conll(path):
    sentences = []
    words, tags = [], []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            if words:
                sentences.append((words, tags))
                words, tags = [], []
            continue
        word, tag = line.rsplit(maxsplit=1)
        words.append(word)
        tags.append(tag)
    if words:
        sentences.append((words, tags))
    return sentences


def collate_fn(items):
    inputs = [item["input_ids"] for item in items]
    labels = [item["label"] for item in items]
    lengths = torch.tensor([len(ids) for ids in inputs], dtype=torch.long)
    return {"input_ids": pad_sequence(inputs, batch_first=True, padding_value=0), "lengths": lengths, "label": pad_sequence(labels, batch_first=True, padding_value=-100)}


class NERVocab:
    def __init__(self, train_path):
        data = read_conll(train_path)
        words = sorted({word for sentence, tags in data for word in sentence})
        tags = sorted({tag for sentence, tags in data for tag in tags})
        self.w2i = {"<pad>": 0, "<unk>": 1}
        self.w2i.update({word: index for index, word in enumerate([word for word in words if word not in self.w2i], start=2)})
        self.l2i = {tag: index for index, tag in enumerate(tags)}
        self.i2l = {index: tag for tag, index in self.l2i.items()}

    @property
    def len(self):
        return len(self.w2i)

    @property
    def n_labels(self):
        return len(self.l2i)

    @property
    def pad_idx(self):
        return 0

    def encode_words(self, words):
        return torch.tensor([self.w2i.get(word, 1) for word in words], dtype=torch.long)

    def encode_tags(self, tags):
        return torch.tensor([self.l2i[tag] for tag in tags], dtype=torch.long)


class PhoNER(Dataset):
    def __init__(self, path, vocab):
        self.data = read_conll(path)
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        words, tags = self.data[index]
        return {"input_ids": self.vocab.encode_words(words), "label": self.vocab.encode_tags(tags)}
