import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_utils.uit_vsfc import UIT_VSFC, Vocab, collate_fn as classification_batch
from data_utils.phoner import PhoNER, NERVocab, collate_fn as ner_batch
from models.classifiers import LstmModel, GruModel
from models.taggers import BiLstmModel
from training.trainer import Trainer


ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_loaders(data_dir, task, batch_size=32, label_key="topic", limit=None, seed=42):
    data_dir = Path(data_dir)
    if task == "ner":
        paths = [data_dir / f"{split}_word.conll" for split in ["train", "dev", "test"]]
        vocab = NERVocab(paths[0])
        datasets = [PhoNER(path, vocab) for path in paths]
        collate = ner_batch
    else:
        paths = [data_dir / f"UIT-VSFC-{split}.json" for split in ["train", "dev", "test"]]
        vocab = Vocab(paths[0], label_key=label_key)
        datasets = [UIT_VSFC(path, vocab) for path in paths]
        collate = classification_batch
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        datasets = [Subset(dataset, range(min(limit, len(dataset)))) for dataset in datasets]
    generator = torch.Generator().manual_seed(seed)
    loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=index == 0, collate_fn=collate, generator=generator if index == 0 else None) for index, dataset in enumerate(datasets)]
    return vocab, loaders


def run_assignment(assignment, data_dir=None, output_dir=None, epochs=30, batch_size=32, learning_rate=0.001, patience=5, seed=42, label_key="topic", device=None, limit=None):
    if assignment not in [1, 2, 3]:
        raise ValueError("assignment must be 1, 2 or 3")
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    task = "ner" if assignment == 3 else "classification"
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "datasets" / ("PhoNER" if assignment == 3 else "UIT-VSFC")
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "outputs" / f"bai{assignment}"
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab, loaders = create_loaders(data_dir, task, batch_size, label_key, limit, seed)
    train_loader, dev_loader, test_loader = loaders
    model_class = {1: LstmModel, 2: GruModel, 3: BiLstmModel}[assignment]
    model = model_class(vocab_size=vocab.len, hidden_size=256, n_layer=5, n_labels=vocab.n_labels, padding_idx=vocab.pad_idx)
    print(f"{model_class.__name__} | device: {device} | train/dev/test: {[len(loader.dataset) for loader in loaders]}", flush=True)
    trainer = Trainer(model, vocab, task, device, learning_rate)
    best = trainer.fit(train_loader, dev_loader, output_dir, epochs, patience)
    test = trainer.evaluate(test_loader)
    print(f"Test F1: {test['f1']:.4f}")
    print(test["report"])
    config = {"assignment": assignment, "hidden_size": 256, "n_layer": 5, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "patience": patience, "seed": seed, "label_key": label_key if task == "classification" else None, "limit": limit}
    result = {"config": config, **best, "test": test, "history": trainer.history}
    (output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    vocab_data = {"words": vocab.w2i, "labels": vocab.l2i, "label_key": config["label_key"]}
    (output_dir / "vocab.json").write_text(json.dumps(vocab_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return trainer, result
