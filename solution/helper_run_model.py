import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


###############################################
# Funktionen zum Anwenden vom ML-Modell       #
###############################################

# Funktion, um Bilder zu verarbeiten
def process_image(image_path, transform, model):
    # Bild einlesen
    image = Image.open(image_path).convert('RGB')

    # Speichere die Originalgröße des Bildes
    original_size = image.size

    # Bild transformieren
    input_tensor = transform(image).unsqueeze(0)  # Batch-Dimension hinzufügen

    # Bild durch das Modell laufen lassen
    with torch.no_grad():
        output = model(input_tensor)  # Modell auf das Bild anwenden

    # Binärbild erstellen (z. B. mit Thresholding)
    output = output.squeeze(0).cpu().numpy()  # Vom Tensor in NumPy umwandeln
    binary_output = (output > 0.5).astype(np.uint8) * 255  # Thresholding auf 0.5 (Anpassbar)

    # Binärbild zurück auf die Originalgröße skalieren
    binary_image = Image.fromarray(binary_output[0])  # Nur das erste Kanalbild, falls RGB
    binary_image = binary_image.resize(original_size, Image.NEAREST)  # Resize auf Originalgröße

    return binary_image


def run_model(part_image_path, model):
    # Transformierungen für das Bild
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Passe die Größe des Bildes an
        transforms.ToTensor(),  # Konvertiere das Bild zu einem Tensor
    ])

    mask = process_image(part_image_path, transform, model)

    return mask