import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet


###############################################
# Klassen für Encoder, UNet und Loss-Funktion #
###############################################

# Definiere das Modell: EfficientNet-B7 + U-Net
class UNetWithEfficientNet(nn.Module):
    def __init__(self):
        super(UNetWithEfficientNet, self).__init__()
        # Lade EfficientNet-B7 als Backbone
        self.encoder = EfficientNet.from_pretrained('efficientnet-b7')

        # U-Net Decoder (Beispielweise könnte das ein einfacher Decoder sein, hier vereinfacht dargestellt)
        # Für eine vollständige U-Net-Architektur müsste der Decoder genau angepasst werden
        self.decoder = nn.Sequential(
            nn.Conv2d(2560, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)  # Ausgabe auf ein Kanal für das Binärbild
        )

        # Clamp Layer: Wird verwendet, um die Ausgaben in einen bestimmten Bereich zu setzen
        self.clamp = nn.Hardtanh(min_val=0.0, max_val=1.0)

    def forward(self, x):
        # Encoder (EfficientNet)
        x = self.encoder.extract_features(x)  # Features extrahieren

        # Decoder (hier stark vereinfacht)
        x = self.decoder(x)

        # Clamp die Ausgaben auf den Bereich [0, 1] (für die Segmentierung)
        x = self.clamp(x)

        return x


# Jaccard-Loss Funktion
class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, output, target):
        # Binarisiere die Ausgabe und das Ziel (z. B. 0 oder 1)
        output = output > 0.5
        target = target > 0.5

        intersection = (output & target).float().sum((1, 2, 3))  # Schnittmenge
        union = (output | target).float().sum((1, 2, 3))  # Vereinigung

        # Berechne den Jaccard-Index und dann den Verlust
        jac = (intersection + 1e-6) / (union + 1e-6)  # Addiere kleinen Wert, um Division durch 0 zu vermeiden
        return 1 - jac.mean()  # Wir wollen den Verlust minimieren (also 1 - Jaccard-Index)


def load_model(modell_path):
    # Modell laden
    checkpoint = torch.load(modell_path, map_location=torch.device('cpu'))

    # Modell definieren
    model = smp.Unet(encoder_name='efficientnet-b7', encoder_weights="imagenet", classes=1, activation='sigmoid')

    # Modell-Dict
    model_dict = model.state_dict()

    # Checkpoint-Dict
    checkpoint_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Nur die Schichten laden, die im Modell existieren
    pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}

    # Die geladenen Schichten in das Modell übertragen
    model_dict.update(pretrained_dict)

    # Modell laden (mit Strict=False, damit es die fehlenden Schichten überspringt)
    model.load_state_dict(model_dict, strict=False)

    # Modell auf eval setzen
    model.eval()

    # Transformierungen für das Bild
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Passe die Größe des Bildes an
        transforms.ToTensor(),  # Konvertiere das Bild zu einem Tensor
    ])

    return model