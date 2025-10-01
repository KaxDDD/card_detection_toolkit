# Pasos de Integración de la IA para `card_detection.py`

**Objetivo:** Reemplazar la lógica de detección basada en colores en `pyclashbot/bot/card_detection.py` con el nuevo modelo basado en una CNN.

---

### Paso 1: Añadir Imports Requeridos

Asegúrate de que los siguientes imports están presentes al principio del archivo `pyclashbot/bot/card_detection.py`:

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

### Paso 2: Borrar Código Obsoleto Basado en Colores

Elimina las siguientes variables y funciones del archivo, ya que pertenecen al sistema antiguo:

-   `card_color_data` (el diccionario gigante)
-   `COLORS`
-   `COLORS_ARRAY`
-   `COLORS_KEYS`
-   `calculate_offset()`
-   `find_closest_card()`
-   `make_pixel_dict_from_color_list()`
-   `color_from_pixel()`
-   `get_corner_pixels()`
-   `get_all_pixel_data()`

### Paso 3: Insertar Lógica del Modelo de IA y Predicción

Inserta el siguiente bloque de código en el archivo, por ejemplo, después de los imports iniciales y las constantes de configuración.

```python
# --- Configuración del Modelo y Datos ---
MODEL_PATH = "card_detection_model.pt" # O la ruta absoluta al archivo
CLASS_MAP_PATH = "card_class_map.json" # O la ruta absoluta al archivo
IMG_SIZE = (54, 66)

# --- Definición de la CNN ---
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

# --- Carga e Inicialización del Modelo ---
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_MAP_PATH):
        print(f"Modelo o mapa de clases no encontrado. Ejecuta train.py.")
        return None, None
    with open(CLASS_MAP_PATH, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    model = CardClassifier(num_classes=len(class_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("Modelo de detección de cartas cargado.")
    return model, idx_to_class

model, idx_to_class = load_model()

# --- Transformación de Imagen ---
transformation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Función de Predicción ---
def predict_card(card_image):
    if model is None: return "UNKNOWN"
    image_tensor = transformation(card_image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
    return idx_to_class.get(predicted_idx.item(), "UNKNOWN")
```

### Paso 4: Reemplazar la Función `identify_hand_cards`

Borra la función `identify_hand_cards` existente y reemplázala con esta nueva implementación:

```python
def identify_hand_cards(emulator, card_index):
    """Toma una captura, recorta la carta y la identifica usando el modelo CNN."""
    screenshot = emulator.screenshot()
    if screenshot is None:
        print("Error al tomar la captura para identificar la carta.")
        return "UNKNOWN"

    # Recorta la imagen de la carta de la captura
    x, y = toplefts[card_index]
    card_image = screenshot[y:y+TOTAL_HEIGHT, x:x+TOTAL_WIDTH]

    # Predice el nombre de la carta
    return predict_card(card_image)
```

**La integración está completa después de estos pasos.**
