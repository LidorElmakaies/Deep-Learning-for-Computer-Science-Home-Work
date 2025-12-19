import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import copy
from pathlib import Path
import time
from datetime import datetime
from utils import save_checkpoint

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.palm_disease_detector import PalmDiseaseDetector
from data.dataset import get_dataloaders
from train import train_epoch, validate


def main():
    # Default values
    data_dir = "data"
    epochs = 50
    batch_size = 32
    lr = 1e-4
    freeze_backbone = True  # Set to False for full fine-tuning

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Load data
    print("Loading datasets...")
    train_loader, val_loader, _ = get_dataloaders(
        data_root=data_dir,
        batch_size=batch_size,
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    # Require validation data
    if val_loader is None:
        raise ValueError(
            "No validation data found. Validation data is required for training."
        )

    # Create model
    print("Creating model...")
    model = PalmDiseaseDetector(freeze_backbone=freeze_backbone)
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping
    best_val_acc = 0.0
    best_train_acc = 0.0  # Track best training accuracy for tie-breaking
    best_epoch = 0
    best_val_loss = float("inf")
    best_model_state = None
    best_optimizer_state = None
    patience = 10  # Check for improvement every X epochs
    has_improvement_in_current_run = (
        False  # Track if there was improvement in current patience window
    )

    print("Starting Training")

    # Training loop - can continue beyond initial epochs if improvement detected
    epoch = 0
    should_stop = False

    while not should_stop:
        epoch_start_time = time.time()

        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        epoch_time = time.time() - epoch_start_time

        # Print summary
        print(f"\n[EPOCH {epoch+1}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s\n")

        # Track best model state
        should_update_best = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_epoch = epoch
            best_val_loss = val_loss
            should_update_best = True
        elif val_acc == best_val_acc:
            # When validation accuracy is equal, check if training accuracy is better
            if train_acc > best_train_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch
                best_val_loss = val_loss
                should_update_best = True

        if should_update_best:
            # Save the best model state for later (deep copy to avoid reference issues)
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            has_improvement_in_current_run = True
            print(
                f"New best validation accuracy: {val_acc:.2f}% (Train Acc: {train_acc:.2f}%)\n"
            )

        # Check for early stopping every X epochs
        if (epoch + 1) % patience == 0:
            if has_improvement_in_current_run:
                # Improvement detected in this patience window - continue training
                has_improvement_in_current_run = False  # Reset for next window
                print(
                    f"Improvement detected at epoch {epoch+1} (best: {best_val_acc:.2f}%). Continuing training.\n"
                )
            else:
                # No improvement in last X epochs - stop training
                print(
                    f"No improvement in validation accuracy over the last {patience} epochs. Best was {best_val_acc:.2f}%. Stopping."
                )
                should_stop = True

        epoch += 1

    print("Training Completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Training Accuracy: {best_train_acc:.2f}%")

    # Save the best model at the end with timestamp and metrics in filename
    if best_model_state is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"best_model_{timestamp}_val{best_val_acc:.2f}_train{best_train_acc:.2f}.pth"

        # Create a temporary model to save the checkpoint
        temp_model = PalmDiseaseDetector(freeze_backbone=freeze_backbone)
        temp_model.load_state_dict(best_model_state)
        temp_optimizer = optim.Adam(temp_model.parameters(), lr=lr)
        temp_optimizer.load_state_dict(best_optimizer_state)

        save_checkpoint(
            temp_model,
            temp_optimizer,
            best_epoch,
            best_val_loss,
            best_val_acc,
            checkpoint_filename,
        )
        print(f"Best model saved: {checkpoint_filename}")
    else:
        print("No best model found to save.")


if __name__ == "__main__":
    main()
