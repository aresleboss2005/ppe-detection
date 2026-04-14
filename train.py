from ultralytics import YOLO
import os

DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.yaml")

BEST_MODEL  = os.path.join("runs", "detect", "ppe_model5", "weights", "best.pt")
START_MODEL = BEST_MODEL if os.path.exists(BEST_MODEL) else "yolov8n.pt"

print(f"Départ depuis : {START_MODEL}")
model = YOLO(START_MODEL)
 
results = model.train(
    data=DATA_YAML,
    epochs=20,        # augmenté à 50
    imgsz=640,
    batch=8,
    name="ppe_model",
    patience=10,
    device='cpu',         # GPU. Changer en 'cpu' si pas de GPU
)

best = os.path.join(str(results.save_dir), "weights", "best.pt")
print(f"\n Entrainement termine ! Meilleur modele : {best}")