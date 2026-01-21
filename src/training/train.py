print("Starting training script...")

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = current_file.parent.parent
sys.path.append(str(src_path))

print(f"Project root set to: {project_root}")
print(f"Source path added to sys.path: {src_path}")

from models.alexnet import get_covid_alexnet
from models.densenet import get_lession_densenet
from dataset import get_data_loaders

def train_model(model_type = "alexnet"):
    DATA_DIR = project_root / "data" / "processed"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    NUM_EPOCHS = 12
    LEARNING_RATE = 0.0001

    print(f"Using device: {DEVICE}")
    print(f"Preparing training model:{model_type.upper()}")
    print(f"Checking data directory at: {DATA_DIR}")

    if not DATA_DIR.exists():
        print(f"Data directory {DATA_DIR} does not exist. Please preprocess the data first.")
        return

    try: 
        if not (DATA_DIR / "train").exists():
            print(f"Training data not found in {DATA_DIR / 'train'}. Please check the dataset.")
            return
        train_loader, val_loader, class_names = get_data_loaders(str(DATA_DIR), BATCH_SIZE)
        print (f"Classes found: {class_names}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    num_classes = len(class_names)
    if model_type == "alexnet":
        model = get_covid_alexnet(num_classes = num_classes).to(DEVICE)
    elif model_type == "densenet":
        model = get_lession_densenet(num_classes = num_classes).to(DEVICE)
    else:
        print(f"Model type {model_type} not recognized. Use 'alexnet' or 'densenet'.")
        return

    criterion = nn.CrossEntropyLoss()
    if model_type == "alexnet":
        optimizer = optim.Adam([
            {'params': model.features.parameters(), 'lr': LEARNING_RATE},
            {'params': model.classifier.parameters(), 'lr': 0.0002}
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    print ("\nStarting training loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}: ", end="")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 2 == 0:
                print(f".", end="", flush=True)

        epoch_acc = 100 * correct / total
        print(f"Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {epoch_acc:.2f}%")

        save_dir = project_root / "models" / "pytorch_checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / f"{model_type}_epoch_{epoch + 1}.pth")
    
    print(f"\nTraining completed. Models saved in {save_dir}")

if __name__ == "__main__":
    print("Training AlexNet...")
    train_model(model_type='alexnet')
    
    print("\nTraining DenseNet...")
    train_model(model_type='densenet')