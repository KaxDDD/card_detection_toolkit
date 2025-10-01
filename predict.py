
import argparse
import json
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# --- Configuration ---
MODEL_PATH = "card_detection_model.pt"
CLASS_MAP_PATH = "card_class_map.json"
IMG_SIZE = (54, 66)

# --- CNN Model Definition (must match the training script) ---
class CardClassifier(nn.Module):
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE[0] // 8) * (IMG_SIZE[1] // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.classifier(x)
        return x

# --- Main Prediction Logic ---
def main():
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description='Predict a Clash Royale card from an image.')
    parser.add_argument('image_path', type=str, help='Path to the card image file.')
    args = parser.parse_args()

    # --- Validate Inputs ---
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at '{args.image_path}'")
        return

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_MAP_PATH):
        print(f"Error: Model ('{MODEL_PATH}') or class map ('{CLASS_MAP_PATH}') not found.")
        print("Please make sure you are in the 'card_detection_toolkit' directory and have run train.py first.")
        return

    # --- Load Model and Class Map ---
    try:
        with open(CLASS_MAP_PATH, 'r') as f:
            class_to_idx = json.load(f)
        
        idx_to_class = {i: name for name, i in class_to_idx.items()}
        num_classes = len(class_to_idx)

        model = CardClassifier(num_classes=num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Image Transformation ---
    transformation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- Load and Predict ---
    try:
        image = cv2.imread(args.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_tensor = transformation(image).unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        card_name = idx_to_class.get(predicted_idx.item(), "UNKNOWN")
        
        print(f"Predicted Card: {card_name}")
        print(f"Confidence: {confidence.item() * 100:.2f}%")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
