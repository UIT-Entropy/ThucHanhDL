import json
from pathlib import Path

import torch
from torch import nn

from training.metrics import classification_metrics, ner_metrics


class Trainer:
    def __init__(self, model, vocab, task, device, learning_rate=0.001):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.vocab = vocab
        self.task = task
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.history = {"train_loss": [], "dev_loss": [], "dev_f1": []}

    def train_epoch(self, loader):
        self.model.train()
        total_loss, total_items = 0.0, 0
        for batch in loader:
            inputs = batch["input_ids"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(inputs, batch["lengths"])
            loss = self.loss_fn(logits.reshape(-1, self.vocab.n_labels), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            count = int((labels != -100).sum().item())
            total_loss += loss.item() * count
            total_items += count
        return total_loss / total_items

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, total_items = 0.0, 0
        all_labels, all_predictions = [], []
        for batch in loader:
            inputs = batch["input_ids"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = self.model(inputs, batch["lengths"])
            loss = self.loss_fn(logits.reshape(-1, self.vocab.n_labels), labels.reshape(-1))
            predictions = logits.argmax(dim=-1).cpu()
            count = int((labels != -100).sum().item())
            total_loss += loss.item() * count
            total_items += count
            if self.task == "ner":
                for row, length in enumerate(batch["lengths"].tolist()):
                    all_labels.append([self.vocab.i2l[index] for index in batch["label"][row, :length].tolist()])
                    all_predictions.append([self.vocab.i2l[index] for index in predictions[row, :length].tolist()])
            else:
                all_labels.extend(batch["label"].tolist())
                all_predictions.extend(predictions.tolist())
        result = ner_metrics(all_labels, all_predictions) if self.task == "ner" else classification_metrics(all_labels, all_predictions, self.vocab)
        result["loss"] = total_loss / total_items
        return result

    def fit(self, train_loader, dev_loader, output_dir, epochs=30, patience=5):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "dev_loss": [], "dev_f1": []}
        best_f1, best_epoch, wait = -1.0, 0, 0
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            dev = self.evaluate(dev_loader)
            self.history["train_loss"].append(train_loss)
            self.history["dev_loss"].append(dev["loss"])
            self.history["dev_f1"].append(dev["f1"])
            print(f"Epoch {epoch}/{epochs} | train loss: {train_loss:.4f} | dev loss: {dev['loss']:.4f} | dev F1: {dev['f1']:.4f}", flush=True)
            if dev["f1"] > best_f1:
                best_f1, best_epoch, wait = dev["f1"], epoch, 0
                torch.save(self.model.state_dict(), output_dir / "best_model.pt")
            else:
                wait += 1
            (output_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            if patience and wait >= patience:
                print("Early stopping", flush=True)
                break
        state = torch.load(output_dir / "best_model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        return {"best_epoch": best_epoch, "dev_f1": best_f1}
