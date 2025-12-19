import torch
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()  # Training mode (allows gradient)

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")  # Progress bar

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).long()  # requires for CrossEntropyLoss

        optimizer.zero_grad()  # Reset gradients from previous iteration

        outputs = model(images)  # shape [batch_size, 2] - logits [healthy, sick]

        loss = criterion(outputs, labels)  # Calculate Loss (CrossEntropyLoss)

        # backwards propagation (calculate gradients)
        loss.backward()  # Calculate gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()

        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()  # count correct predictions
        # total number of images that were processed in this batch (size of the batch)
        total += labels.size(0)

        # Update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100 * (correct / total)
        pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"})

    # Final averages
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, epoch):
    model.eval()  # Evaluation mode (doesn't allow gradient - saves memory)

    running_loss = 0.0  # sum of losses for this epoch
    correct = 0  # count correct predictions
    total = 0  # count total samples processed

    # Don't calculate gradients in validation
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).long()

            # Forward pass only (no backward)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculate Loss (CrossEntropyLoss)

            # Statistics
            running_loss += loss.item()  # sum of losses for this epoch

            # Get predicted class (0=healthy, 1=sick)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)  # count total samples processed

            # Update progress bar
            current_loss = running_loss / len(val_loader)
            current_acc = 100 * correct / total
            pbar.set_postfix(
                {"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"}
            )

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy
