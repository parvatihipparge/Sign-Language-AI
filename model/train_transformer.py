import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, random_split
from model import LandmarkTransformer


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 126
NUM_CLASSES = 37

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "dataset")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "model", "checkpoint.pth")


# Load dataset
print("\nLoading dataset...")

landmarks = np.load(os.path.join(DATA_DIR, "landmarks.npy"))
labels = np.load(os.path.join(DATA_DIR, "labels.npy"))

print("Landmarks:", landmarks.shape)
print("Labels:", labels.shape)

if landmarks.shape != (7400, INPUT_DIM):
    raise ValueError(f"Unexpected landmark shape: {landmarks.shape}")

if labels.shape != (7400,):
    raise ValueError(f"Unexpected label shape: {labels.shape}")

if not np.isfinite(landmarks).all():
    raise ValueError("Dataset contains invalid values.")

if set(np.unique(labels)) != set(range(NUM_CLASSES)):
    raise ValueError("Dataset must contain all 37 gesture classes.")

# Convert to tensors
X = torch.tensor(landmarks, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(X, y)

# 80% training and 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training samples:", train_size)
print("Validation samples:", val_size)


# Initialize model
model = LandmarkTransformer(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_accuracy = 0.0


# Training
print("\nTraining started...")
print("Device:", DEVICE)

for epoch in range(NUM_EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += targets.size(0)

        total_loss += loss.item()

    train_accuracy = 100 * correct / total
    train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == targets).sum().item()
            val_total += targets.size(0)

            val_loss += loss.item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Accuracy: {train_accuracy:.2f}% "
        f"| Val Loss: {avg_val_loss:.4f} "
        f"| Val Accuracy: {val_accuracy:.2f}%"
    )

    # Save best model
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        print("Best model checkpoint saved!")


print("\nTraining completed!")
print(f"Best validation accuracy: {best_accuracy:.2f}%")
print("Model saved at:", CHECKPOINT_PATH)