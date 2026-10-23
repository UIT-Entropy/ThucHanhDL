import torch
from torch import nn


def fit(model, dataloader, device, epochs=10, learning_rate=0.001):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(x_batch)
                loss = loss_fn(output, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y_batch)
            correct += (output.argmax(dim=1) == y_batch).sum().item()
        loss = total_loss / len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        history.append({"loss": loss, "accuracy": accuracy})
        print(f"Epoch {epoch + 1}/{epochs} | loss: {loss:.4f} | accuracy: {accuracy:.4f}", flush=True)
    return history


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch, _ in dataloader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(x_batch.to(device, non_blocking=True))
            predictions.append(output.argmax(dim=1).cpu())
    return torch.cat(predictions).numpy()
