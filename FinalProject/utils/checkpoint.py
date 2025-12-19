import torch
import os

# Default directories
SAVE_DIR = "checkpoints"
# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename):
    filepath = os.path.join(SAVE_DIR, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),  # All model weights
        "optimizer_state_dict": optimizer.state_dict(),  # Optimizer state (learning rate, etc.)
        "loss": loss,
        "accuracy": accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Checkpoint loaded from {filepath}, resuming from epoch {start_epoch}")
    return start_epoch
