import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

# -------------------- CONFIG --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pth")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

# -------------------- LOAD DATA --------------------
print("Loading dataset...")
try:
    landmarks = np.load(os.path.join(DATA_DIR, "landmarks.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    print(f"Dataset loaded successfully! Samples: {len(landmarks)}")
    
    # Automatically determine number of classes from the data
    NUM_CLASSES = int(labels.max() + 1)
    print(f"Number of classes detected: {NUM_CLASSES}")
    print(f"Label range: {labels.min()} to {labels.max()}")
    
except FileNotFoundError:
    print("Error: Dataset files not found!")
    print(f"Looking for files in: {DATA_DIR}")
    print("Please make sure landmarks.npy and labels.npy exist in the dataset folder")
    exit(1)

# Convert to tensors
X = torch.FloatTensor(landmarks)
y = torch.LongTensor(labels)

# Create data loader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------- MODEL --------------------
class LandmarkTransformer(nn.Module):
    def __init__(self, input_dim=63, embed_dim=64, num_heads=4, num_layers=2, num_classes=36):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

# -------------------- TRAINING --------------------
print(f"Training on {DEVICE}")
model = LandmarkTransformer(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print(f"\nStarting training for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        total_loss += loss.item()
    
    # Print epoch statistics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
    
    # Save checkpoint every 10 epochs and at the end
    if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"✓ Checkpoint saved at epoch {epoch+1}")

print("\n✓ Training completed successfully!")
print(f"Final model saved to: {CHECKPOINT_PATH}")