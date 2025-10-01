# AI Card Detection Toolkit

## Overview

This toolkit contains everything you need to capture images, train the AI model, and test it.

## Folder Contents

-   `dataset/`: A folder containing all the card images used to train the AI.
-   `capture.py`: A script to capture new card images from the game.
-   `train.py`: The script that trains the AI using the images from the `dataset/` folder.
-   `predict.py`: A utility to test the AI with a single image and see its prediction.
-   `card_detection_model.pt`: The AI's "brain". This file is updated every time you run `train.py`.
-   `card_class_map.json`: A file that helps the AI know the name for each prediction.

---

## Guide for Non-Experts

Imagine the AI is a new student. Your job is to be its teacher.

### Task 1: Add More Examples (Teaching)

If the AI makes a mistake or doesn't know a card, you need to give it more examples.

1.  **Open a terminal as Administrator:**
    *   Go to the Start Menu, type `cmd`, right-click on "Command Prompt", and select "Run as administrator".

2.  **Navigate to this folder:** In the terminal, type (you can copy and paste):
    ```sh
    cd D:\CRBOT\card_detection_toolkit
    ```

3.  **Start capture mode:**
    ```sh
    python capture.py
    ```

4.  **Capture in-game images:** With Clash Royale open, press `Enter` in the terminal to save images of the 4 cards on screen. Do this multiple times, especially with cards the AI doesn't recognize well and their grayed-out (inactive) versions.

5.  **Label the new "photos":**
    *   The new images will appear in a folder named `card_capture_output` (inside `card_detection_toolkit`).
    *   Rename each image with the card's name and a number (e.g., `giant_23.png`, `giant_24.png`).
    *   Move these renamed images into the `dataset/` folder.

### Task 2: Study and Learn (Training)

After adding new examples, the "student" (the AI) needs time to study.

1.  **Open a terminal** in this folder (`D:\CRBOT\card_detection_toolkit`).
2.  **Start the training:**
    ```sh
    python train.py
    ```
3.  Wait for the process to finish. You will see the "Validation Accuracy" (the student's grade) improve. This may take several minutes.

### Task 3: Test the Student (Predicting)

Want to see if the AI has learned to recognize a specific image?

1.  **Open a terminal** in this folder.
2.  **Ask for a prediction:**
    ```sh
    python predict.py "path\\to\\your\\image.png"
    ```
    For example, to test with an image of a giant from your `dataset`:
    ```sh
    python predict.py "dataset\giant_10.png"
    ```
3.  The script will tell you which card it thinks it is and with what level of confidence.

---

By repeating steps 1 and 2 (adding more images and re-training), you will make the AI more and more accurate over time.

---

## For Developers: How to Integrate This Model

These are the instructions to replace the original, color-based detection system in `pyclashbot/bot/card_detection.py` with this new AI model.

### Step 1: Add New Imports

At the top of `pyclashbot/bot/card_detection.py`, add the necessary imports for PyTorch and json:

```python
import json
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
```

### Step 2: Define and Load the AI Model

You need to add the Python code that defines the neural network architecture and loads the trained files.

1.  **Copy the `CardClassifier` class** from `train.py` into `card_detection.py`.
2.  **Add code to load the model** (`.pt` file) and the class map (`.json` file) at startup. 
3.  **Create a `predict_card` function** that takes an image, applies the correct transformations, and returns the predicted card name.

Here is a code block that accomplishes these three points. You can place this below the imports:

```python
# --- Model and Data Configuration ---
MODEL_PATH = "card_detection_model.pt" # Or the absolute path to it
CLASS_MAP_PATH = "card_class_map.json" # Or the absolute path to it
IMG_SIZE = (54, 66)

# --- CNN Model Definition (must match the training script) ---
class CardClassifier(nn.Module):
    # ... (copy the class definition here) ...

# --- Model Loading and Initialization ---
def load_model():
    # ... (copy the load_model function here) ...

model, idx_to_class = load_model()

# Define the image transformation pipeline
transformation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Card Identification Function ---
def predict_card(card_image):
    # ... (copy the predict_card function here) ...
```

### Step 3: Replace the Core Detection Logic

The main change happens in the `identify_hand_cards` function. You need to replace its body entirely.

**Delete the old code:**
```python
def identify_hand_cards(emulator, card_index):
    color_chosen_card = get_all_pixel_data(emulator, card_index)
    card_name = find_closest_card(color_chosen_card)
    # ... (and the rest of the old logic)
    return card_name
```

**Add the new code:**
```python
def identify_hand_cards(emulator, card_index):
    """Takes a screenshot, crops the card, and identifies it using the CNN model."""
    screenshot = emulator.screenshot()
    if screenshot is None:
        print("Error taking screenshot for card identification.")
        return "UNKNOWN"

    # Crop the card image from the screenshot
    x, y = toplefts[card_index] # toplefts is a global constant
    card_image = screenshot[y:y+TOTAL_HEIGHT, x:x+TOTAL_WIDTH]

    # Predict the card name
    return predict_card(card_image)
```

### Step 4: Clean Up Old Code

Finally, you can delete all the code related to the old color-based system, as it is no longer used. This includes:

-   The massive `card_color_data` dictionary.
-   The `COLORS`, `COLORS_ARRAY`, and `COLORS_KEYS` constants.
-   The functions: `calculate_offset`, `find_closest_card`, `make_pixel_dict_from_color_list`, `color_from_pixel`, and `get_all_pixel_data`.

This will leave you with a much cleaner and more effective card detection file.

