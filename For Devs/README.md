# Kit de Herramientas para la Detección de Cartas por IA

Este kit de herramientas contiene todo lo que necesitas para capturar imágenes, entrenar el modelo de IA y probarlo.

## Contenido de la Carpeta

-   `dataset/`: Una carpeta que contiene todas las imágenes de cartas que se usarán para entrenar a la IA.
-   `capture.py`: Un script para capturar nuevas imágenes de cartas desde el juego.
-   `train.py`: El script que entrena a la IA usando las imágenes de la carpeta `dataset/`.
-   `predict.py`: Una utilidad para probar la IA con una sola imagen y ver qué predice.
-   `card_detection_model.pt`: El "cerebro" de la IA. Este archivo se actualiza cada vez que ejecutas `train.py`.
-   `card_class_map.json`: Un archivo que ayuda a la IA a saber el nombre de cada carta.

---

## Guía para No Expertos

Imagina que la IA es un estudiante nuevo. Tu trabajo es ser su profesor.

### Tarea 1: Añadir más ejemplos (Enseñar)

Si la IA se equivoca o no conoce una carta, necesitas darle más ejemplos.

1.  **Abre una terminal como Administrador:**
    *   Ve al Menú Inicio, escribe `cmd`, haz clic derecho en "Símbolo del sistema" y selecciona "Ejecutar como administrador".

2.  **Navega a esta carpeta:** En la terminal, escribe (puedes copiar y pegar):
    ```sh
    cd D:\CRBOT\card_detection_toolkit
    ```

3.  **Inicia el modo de captura:**
    ```sh
    python capture.py
    ```

4.  **Captura imágenes en el juego:** Con Clash Royale abierto, presiona `Enter` en la terminal para guardar imágenes de las 4 cartas en pantalla. Haz esto varias veces, especialmente con las cartas que la IA no reconoce bien y con sus versiones en gris (cuando no hay elixir).

5.  **Etiqueta las nuevas "fotos":**
    *   Las imágenes nuevas aparecerán en una carpeta llamada `card_capture_output` (dentro de `card_detection_toolkit`).
    *   Renombra cada imagen con el nombre de la carta y un número (ej: `gigante_23.png`, `gigante_24.png`).
    *   Mueve estas imágenes ya renombradas a la carpeta `dataset/`.

### Tarea 2: Estudiar y Aprender (Entrenar)

Después de añadir nuevos ejemplos, el "estudiante" (la IA) necesita tiempo para estudiar.

1.  **Abre una terminal** en esta carpeta (`D:\CRBOT\card_detection_toolkit`).
2.  **Inicia el entrenamiento:**
    ```sh
    python train.py
    ```
3.  Espera a que el proceso termine. Verás que el "Validation Accuracy" (la nota del estudiante) mejora. Esto puede tardar varios minutos.

### Tarea 3: Poner a Prueba al Estudiante (Predecir)

¿Quieres ver si la IA ha aprendido a reconocer una imagen específica?

1.  **Abre una terminal** en esta carpeta.
2.  **Pide una predicción:**
    ```sh
    python predict.py "ruta\a\tu\imagen.png"
    ```
    Por ejemplo, para probar con una imagen de un gigante que está en tu `dataset`:
    ```sh
    python predict.py "dataset\giant_10.png"
    ```
3.  El script te dirá qué carta cree que es y con qué nivel de confianza.

---

Repitiendo los pasos 1 y 2 (añadir más imágenes y re-entrenar) es como conseguirás que la IA sea cada vez más y más precisa.

---

## Para Desarrolladores: Cómo Integrar este Modelo

Estas son las instrucciones para reemplazar el sistema de detección original basado en colores en `pyclashbot/bot/card_detection.py` con este nuevo modelo de IA.

### Paso 1: Añadir Nuevos Imports

En la parte superior de `pyclashbot/bot/card_detection.py`, añade los imports necesarios para PyTorch y json:

```python
import json
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
```

### Paso 2: Definir y Cargar el Modelo de IA

Necesitas añadir el código de Python que define la arquitectura de la red neuronal y carga los archivos entrenados.

1.  **Copia la clase `CardClassifier`** desde `train.py` a `card_detection.py`.
2.  **Añade el código para cargar el modelo** (archivo `.pt`) y el mapa de clases (archivo `.json`) al iniciar el script.
3.  **Crea una función `predict_card`** que tome una imagen, le aplique las transformaciones correctas y devuelva el nombre de la carta predicha.

Este es un bloque de código que resume esos tres puntos. Puedes colocarlo debajo de los imports:

```python
# --- Configuración del Modelo y Datos ---
MODEL_PATH = "card_detection_model.pt" # O la ruta absoluta al archivo
CLASS_MAP_PATH = "card_class_map.json" # O la ruta absoluta al archivo
IMG_SIZE = (54, 66)

# --- Definición de la CNN (debe coincidir con el script de entrenamiento) ---
class CardClassifier(nn.Module):
    # ... (copia la definición de la clase aquí) ...

# --- Carga e Inicialización del Modelo ---
def load_model():
    # ... (copia la función load_model aquí) ...

model, idx_to_class = load_model()

# Define las transformaciones de la imagen
transformation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Función de Identificación de Cartas ---
def predict_card(card_image):
    # ... (copia la función predict_card aquí) ...
```

### Paso 3: Reemplazar la Lógica de Detección Principal

El cambio principal ocurre en la función `identify_hand_cards`. Tienes que reemplazar su contenido por completo.

**Borra el código antiguo:**
```python
def identify_hand_cards(emulator, card_index):
    color_chosen_card = get_all_pixel_data(emulator, card_index)
    card_name = find_closest_card(color_chosen_card)
    # ... (y el resto de la lógica antigua)
    return card_name
```

**Añade el nuevo código:**
```python
def identify_hand_cards(emulator, card_index):
    """Toma una captura, recorta la carta y la identifica usando el modelo CNN."""
    screenshot = emulator.screenshot()
    if screenshot is None:
        print("Error al tomar la captura para identificar la carta.")
        return "UNKNOWN"

    # Recorta la imagen de la carta de la captura
    x, y = toplefts[card_index] # toplefts es una constante global
    card_image = screenshot[y:y+TOTAL_HEIGHT, x:x+TOTAL_WIDTH]

    # Predice el nombre de la carta
    return predict_card(card_image)
```

### Paso 4: Limpiar el Código Antiguo

Finalmente, puedes borrar todo el código relacionado con el antiguo sistema de colores, ya que no se usa.

-   El diccionario gigante `card_color_data`.
-   Las constantes `COLORS`, `COLORS_ARRAY`, y `COLORS_KEYS`.
-   Las funciones: `calculate_offset`, `find_closest_card`, `make_pixel_dict_from_color_list`, `color_from_pixel`, y `get_all_pixel_data`.

Esto te dejará con un archivo de detección de cartas mucho más limpio y efectivo.

