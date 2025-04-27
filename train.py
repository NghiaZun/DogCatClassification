import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import numpy as np
import os
import timm
import copy
from model_arch import CatDogClassifier

# Config hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset and Dataloader (for Dog and Cat) 
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize ảnh về kích thước phù hợp với ConvNeXt
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Mean và Std phù hợp với ConvNeXt
                         std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root='/kaggle/input/cat-and-dog/training_set/training_set', transform=transform)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model Architecture
model = CatDogClassifier(num_labels=2)
model = model.to(DEVICE)

# Use BCEWithLogitsLoss for binary classification (chó hoặc mèo)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

# Early Stopping 
best_val_loss = float('inf')
patience = 3
counter = 0
best_model_state = None
os.makedirs("checkpoints", exist_ok=True)

# Evaluation: Calculate Loss and Accuracy
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_logits = []
    total_loss = 0

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)

        loss = criterion(logits.squeeze(1), labels.float())  

        total_loss += loss.item()

        all_logits.append(logits.sigmoid().cpu().numpy().flatten())  # Flatten logits to ensure one dimension
        all_labels.append(labels.cpu().numpy().flatten())  # Flatten labels to ensure one dimension

    all_logits = np.concatenate(all_logits, axis=0)  
    all_labels = np.concatenate(all_labels, axis=0)  

    # Tính accuracy cho phân loại nhị phân
    preds = (all_logits > 0.5).astype(int)  # threshold 0.5 for binary classification
    accuracy = np.mean(preds == all_labels)
    return total_loss / len(dataloader), accuracy

# Training Loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits.squeeze(1), labels.float())  # Remove the extra dimension of logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, f"checkpoints/best_model_epoch{epoch+1}_val{val_loss:.4f}.pth")
        print("Improved Val Loss — model saved.")
    else:
        counter += 1
        print(f"No improvement in val loss. Patience: {counter}/{patience}")

    # Early stopping
    if counter >= patience:
        print("Early stopping triggered.")
        break

# Save the model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/catdog_final.pth")
print("\nTraining complete. Model saved to checkpoints/catdog_final.pth")
