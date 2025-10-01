# AI Integration Steps for `card_detection.py`

**Objective:** Replace the color-based detection logic in `pyclashbot/bot/card_detection.py` with the new CNN-based model.

---

### Step 1: Add Required Imports

Ensure the following imports are present at the top of the file `pyclashbot/bot/card_detection.py`:

```python
import json
import os
import random
import time
import cv2
import numpy
import torch
import torch.nn as nn
from torchvision import transforms
```

### Step 2: Delete Obsolete Color-Based Code

Remove the following variables and functions from the file, as they belong to the old system:

-   `card_color_data` (the large dictionary)
-   `COLORS`
-   `COLORS_ARRAY`
-   `COLORS_KEYS`
-   `calculate_offset()`
-   `find_closest_card()`
-   `make_pixel_dict_from_color_list()`
-   `color_from_pixel()`
-   `get_corner_pixels()`
-   `get_all_pixel_data()`

### Step 3: Insert AI Model and Prediction Logic

Insert the following code block into the file, for example, after the initial imports and configuration constants.

```python
# --- Model and Data Configuration ---
MODEL_PATH = "card_detection_model.pt"
CLASS_MAP_PATH = "card_class_map.json"
IMG_SIZE = (54, 66)

# --- CNN Model Definition ---
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

# --- Model Loading and Initialization ---
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_MAP_PATH):
        print(f"Model or class map not found. Run train.py.")
        return None, None
    with open(CLASS_MAP_PATH, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    model = CardClassifier(num_classes=len(class_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("Card detection model loaded.")
    return model, idx_to_class

model, idx_to_class = load_model()

# --- Image Transformation ---
transformation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Prediction Function ---
def predict_card(card_image):
    if model is None: return "UNKNOWN"
    image_tensor = transformation(card_image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
    return idx_to_class.get(predicted_idx.item(), "UNKNOWN")
```

### Step 4: Replace `identify_hand_cards` Function

Delete the existing `identify_hand_cards` function and replace it with this new implementation:

```python
def identify_hand_cards(emulator, card_index):
    """Takes a screenshot, crops the card, and identifies it using the CNN model."""
    screenshot = emulator.screenshot()
    if screenshot is None:
        print("Error taking screenshot for card identification.")
        return "UNKNOWN"

    # Crop the card image from the screenshot
    x, y = toplefts[card_index]
    card_image = screenshot[y:y+TOTAL_HEIGHT, x:x+TOTAL_WIDTH]

    # Predict the card name
    return predict_card(card_image)
```

**Integration is complete after these steps.**
