
import os
import json
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Configuration ---
DATA_DIR = "dataset"
MODEL_SAVE_PATH = "card_detection_model.pt"
CLASS_MAP_SAVE_PATH = "card_class_map.json"
IMG_SIZE = (54, 66)  # Resize images to a consistent size, respecting aspect ratio
BATCH_SIZE = 16
EPOCHS = 75 # More epochs for a smaller learning rate
LEARNING_RATE = 0.0001 # Smaller learning rate for finer optimization

# --- 1. Custom Dataset Class ---
class CardDataset(Dataset):
    """Custom PyTorch Dataset for loading card images."""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # Load image with OpenCV
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            # Return a dummy image and label if an image is corrupted
            return torch.zeros(3, IMG_SIZE[1], IMG_SIZE[0]), -1 # Note: H, W for tensor

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 2. CNN Model Definition ---
class CardClassifier(nn.Module):
    """A simple Convolutional Neural Network for classifying cards."""
    def __init__(self, num_classes):
        super(CardClassifier, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Adjust the linear layer size based on the new IMG_SIZE
        # New dimensions after 3 maxpools: (54//8, 66//8) -> (6, 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE[0] // 8) * (IMG_SIZE[1] // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.classifier(x)
        return x

def main():
    """Main function to run the training process."""
    print("Starting card detection training process...")

    # --- 3. Load and Prepare Data ---
    print(f"Loading images from: {DATA_DIR}")
    
    # Get all image paths
    try:
        all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(('.png', '.jpg'))]
    except FileNotFoundError:
        print(f"Error: The directory '{DATA_DIR}' was not found.")
        print("Please make sure you have created the 'card_dataset' directory and placed your labeled images inside.")
        return

    if not all_files:
        print(f"Error: No images found in '{DATA_DIR}'.")
        print("Please make sure your labeled images are in the 'card_dataset' directory.")
        return

    # Create robust class mapping from filenames
    def get_class_name_from_file(f):
        name = os.path.splitext(os.path.basename(f))[0]
        name = name.lower().replace('-', '')
        label = name.split('_')[0]
        return label

    class_names = sorted(list(set([get_class_name_from_file(f) for f in all_files])))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    print(f"Found {len(all_files)} images belonging to {num_classes} classes.")
    if num_classes < 2:
        print("Error: Need at least 2 different classes (types of cards) to train the model.")
        return

    # Create the full dataset list (image_path, label_index)
    full_dataset = []
    for f in all_files:
        class_name = get_class_name_from_file(f)
        label = class_to_idx[class_name]
        full_dataset.append((f, label))

    # Split data into training and validation sets
    random.shuffle(full_dataset)
    split_idx = int(len(full_dataset) * (1 - 0.2)) # Re-introducing the 80/20 split
    train_data = full_dataset[:split_idx]
    val_data = full_dataset[split_idx:]

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = CardDataset(train_data, transform=train_transform)
    val_dataset = CardDataset(val_data, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 4. Initialize Model, Loss, and Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CardClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- 5. Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            if -1 in labels:
                continue
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if -1 in labels:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Training Loss: {running_loss/len(train_loader):.4f} | "
              f"Validation Loss: {val_loss/len(val_loader):.4f} | "
              f"Validation Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step()

    print("--- Finished Training ---")

    # --- 6. Save Model and Class Map ---
    print(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Saving class map to {CLASS_MAP_SAVE_PATH}")
    with open(CLASS_MAP_SAVE_PATH, 'w') as f:
        json.dump(class_to_idx, f, indent=4)
        
    print("\nTraining complete!")
    print("You can now integrate the trained model into the bot.")

if __name__ == "__main__":
    main()
