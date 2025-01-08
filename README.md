# Hackathon 2024 - Submission of Group *Zwischenlösung*

Team members:
    - Johannes März
    - Quentin Koplin
    - Robert Maier

## Description
Im ersten Schritt der Lösung wird ein Machine-Vision-Ansatz angewendet, der ein Bild von einem Blechteil in ein Binärbild umwandelt. Dieser Ansatz basiert auf der Bildsegmentierung "Semantic Segmentation". Für die Architektur des Convolutional Neural Networks wurde U-Net gewählt.

// Quentins part (max. 3 Sätze - Insgesamt sollen es so 5 bis 6 sätze sein) //

## How to Run
// How to run your code, e.g., `python solution/main.py path/to/input/tasks.csv output/solutions.csv` //

## ... and other things you want to tell us
# Besonderheiten
- Greifermittelpunkt(**Wichtig**): Die Optimierung der Positionierung erfolgt zum **Bauteilschwerpunkt** und nicht dem Bildmittelpunkt. Da der Schwerpunkt die tatsächliche physikalische Balance und Stabilität des Bauteils berücksichtigt, was präziseres Greifen und sichereren Transport ermöglicht.
- Parameter zur Positio:
    - Randbereich: Definiert den Randbereich den der Greifer mindestens einhalten soll. Der Abstand kann abhängig der Bauteileigenschaften
    - Rechenaufwand der Positionierung: Der Rechenaufwand kann durch die Variable **num_iter** erheblich beeinflusst werden. Dieser bestimmt die Netzauflösung für die Positionierung
 - Warnings und Errors:

- SVGs können auch verarbeitet werden


# Funktionsweise der Positionierung und Scoring


# Modellparameter
Modellarchitektur (U-Net mit EfficientNet-B7 als Encoder):
Aufgrund ihrer bewährten Leistung in Bildsegmentierungsaufgaben wurde die U-Net-Architektur gewählt. Sie ermöglicht die Extraktion sowohl globaler als auch lokaler Merkmale, die für die Bearbeitung von verrauschten und variierenden Bildern essenziell sind. Die Integration des Encoders EfficientNet-B7 erfolgte unter anderem aufgrund seiner hohen Kapazität und Effizienz, die ihn für die Extraktion von Merkmalen aus komplexen und hochdimensionalen Bilddaten prädestinieren. Dies unterstützt die Bewältigung der Bildvariabilität und ermöglicht eine robuste Segmentierung trotz vorhandener Störungen.

Optimizer (RMSprop):
Die Wahl von RMSprop als Optimizer erfolgte aufgrund seiner spezifischen Fähigkeit, insbesondere bei verrauschten Gradienten, eine konsistente und effiziente Optimierung zu gewährleisten. Dies ist von besonderer Relevanz, da bei stark verrauschten und unscharfen Bildern die Gradienten häufig uneinheitlich sind.

Aktivierungsfunktion (Sigmoid):
Die Verwendung der Sigmoid-Aktivierungsfunktion am Ende des Modells fördert die binäre Segmentierung der Labels. Die Anwendung dieser Funktion ermöglicht die präzise Vorhersage von Wahrscheinlichkeiten für die Zugehörigkeit zu einer bestimmten Klasse, was in diesem Fall einen wesentlichen Beitrag zur Kantenerkennung leistet.

Loss-Funktion (JaccardLoss):
Die Jaccard-Loss-Funktion wurde ausgewählt, da sie sich für Segmentierungsaufgaben mit ungleichen Klassenverteilungen als geeignet erwiesen hat. Sie maximiert die Ähnlichkeit zwischen den vorhergesagten Segmenten und den tatsächlichen Labels und gleicht dabei die Auswirkungen von verrauschten oder unvollständigen Label-Definitionen aus.


# Disclaimer
Bei der initialen Ausführung des Modells ist mit einer längeren Ladezeit zu rechnen. Dieser Vorgang umfasst das Laden der Parameter, des Modells und der Funktionen. Bei der Verwendung eines Laptops ist darauf zu achten, dass dieser nicht im Energiesparmodus, sondern an das Netz angeschlossen ist. Andernfalls kann sich die Berechnungszeit signifikant erhöhen.


