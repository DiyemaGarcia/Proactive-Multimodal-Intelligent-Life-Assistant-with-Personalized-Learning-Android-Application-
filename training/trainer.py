"""
Standard SGD training loop.
Used to train source and target models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 60,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    lr_schedule: Optional[List[int]] = None,
    lr_gamma: float = 0.1,
    device: str = "cpu",
    verbose: bool = True,
    save_checkpoints: bool = False,
    checkpoint_dir: str = "checkpoints"
) -> Tuple[nn.Module, Dict]:
    """
    Trains a model with SGD + momentum.

    Args:
        model: PyTorch model to train
        train_loader: training DataLoader
        val_loader: validation DataLoader (optional)
        epochs: number of epochs
        lr: initial learning rate
        momentum: momentum for SGD
        weight_decay: L2 regularization
        lr_schedule: list of epochs to multiply LR by lr_gamma
        lr_gamma: LR decay factor
        device: "cpu" or "cuda"
        verbose: display progress
        save_checkpoints: save model checkpoints
        checkpoint_dir: directory for saving checkpoints

    Returns:
        (trained model, metrics history)
    """
    import os

    device = torch.device(device)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    if lr_schedule is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_schedule, gamma=lr_gamma
        )
    else:
        scheduler = None

    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") \
            if verbose else train_loader

        for batch in iterator:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += x.size(0)

        avg_train_loss = train_loss / train_total
        avg_train_acc = 100.0 * train_correct / train_total
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)

        # ---- Validation ----
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, device=str(device))
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        else:
            val_loss, val_acc = 0.0, 0.0

        if verbose:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

        if scheduler is not None:
            scheduler.step()

        if save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)

    return model, history


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    loss_fn: Optional[nn.Module] = None
) -> Tuple[float, float]:
    """
    Evaluates a model on a DataLoader.

    Args:
        model: PyTorch model
        dataloader: test/validation DataLoader
        device: device
        loss_fn: loss function (CrossEntropy by default)

    Returns:
        (average loss, accuracy in percentage)
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            output = model(x)
            loss = loss_fn(output, y)

            total_loss += loss.item() * x.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def evaluate_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_class,
    model_kwargs: dict,
    dataloader: DataLoader,
    device: str = "cpu"
) -> Tuple[float, float]:
    """
    Evaluates a state_dict directly without instantiating an external model.

    Args:
        state_dict: model parameters
        model_class: model class
        model_kwargs: arguments to instantiate the model
        dataloader: DataLoader
        device: device

    Returns:
        (loss, accuracy)
    """
    model = model_class(**model_kwargs)
    model.load_state_dict(state_dict, strict=False)
    return evaluate_model(model, dataloader, device=device)