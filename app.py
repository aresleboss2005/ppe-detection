# ==============================================================
# app.py  ─  Live PPE detection via webcam
# HOW TO RUN:  python app.py
# QUIT:        press Q
# ==============================================================

import cv2
from ultralytics import YOLO
import os

# ── STEP 1: load the trained model ────────────────────────────────────────
MODEL_PATH = os.path.join("runs", "detect", "ppe_model6", "weights", "best.pt")

if not os.path.exists(MODEL_PATH):
    print("❌ Trained model not found at:", MODEL_PATH)
    print("   → Run train.py first, then try again.")
    exit()

model = YOLO(MODEL_PATH)

# ── STEP 2: read class names straight from the model ──────────────────────
# This is 100% safe: the model learned the class names from data.yaml
# We do NOT hardcode names here → no risk of mismatch
CLASS_NAMES = model.names   # dict like {0: 'Hardhat', 1: 'Mask', ...}
print("✅ Model loaded. Classes:", CLASS_NAMES)

# ── STEP 3: define which class names mean PPE is MISSING ──────────────────
# These must match EXACTLY what is in your data.yaml names list
# (case-sensitive). Check your data.yaml and adjust if needed.
MISSING_PPE = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

# Classes that confirm PPE is being worn correctly
WEARING_PPE = {"Hardhat", "Mask", "Safety Vest", "Gloves"}

# Non-PPE classes we skip to keep the view clean
# Add or remove entries to match your dataset's class names
SKIP_CLASSES = {
    "machinery", "vehicle", "Excavator", "Ladder", "Safety Cone",
    "SUV", "bus", "dump truck", "mini-van", "sedan", "semi",
    "trailer", "truck", "truck and trailer", "van",
    "wheel loader", "fire hydrant",
}

# ── STEP 4: open the webcam ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)   # 0 = default/built-in camera
# change to 1 or 2 if you have multiple cameras

if not cap.isOpened():
    print("❌ Cannot open camera. Check it is connected and not used by another app.")
    exit()

print("📷 Camera open. Press Q to quit.\n")

# ── STEP 5: detection loop ────────────────────────────────────────────────
while True:

    ret, frame = cap.read()   # grab one frame

    if not ret:               # camera disconnected or end of stream
        print("❌ Failed to read frame. Exiting.")
        break

    # Run YOLO on the frame
    # conf=0.45 → only show detections with ≥ 45% confidence
    # verbose=False → suppress per-frame console spam
    results = model.predict(source=frame, conf=0.45, verbose=False)

    for result in results:
        for box in result.boxes:

            # ── get detection info ─────────────────────────────────────
            class_id   = int(box.cls[0])           # index number
            class_name = CLASS_NAMES[class_id]     # human-readable name
            confidence = float(box.conf[0])        # 0.0 → 1.0

            # skip non-PPE objects entirely
            if class_name in SKIP_CLASSES:
                continue

            # bounding box corners
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ── choose color and status label ─────────────────────────
            if class_name in MISSING_PPE:
                color  = (0, 0, 255)      # RED   ❌  PPE missing
                status = "MISSING"

            elif class_name in WEARING_PPE:
                color  = (0, 200, 0)      # GREEN ✅  PPE worn
                status = "OK"

            elif class_name == "Person":
                color  = (0, 165, 255)    # ORANGE 👤  person (PPE unknown)
                status = "Person"

            else:
                color  = (200, 200, 200)  # GRAY  — any other class
                status = class_name

            # ── draw bounding box ──────────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

            # ── build text label  e.g. "NO-Hardhat MISSING 83%" ───────
            label = f"{class_name}  {status}  {confidence:.0%}"

            # measure text size so the background fits correctly
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )

            # filled rectangle behind the text (goes above the box)
            label_y_top = max(y1 - text_h - baseline - 4, 0)
            cv2.rectangle(
                frame,
                (x1, label_y_top),
                (x1 + text_w + 4, y1),
                color, -1              # -1 = filled
            )

            # white text on top of the filled rectangle
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.55,
                color=(255, 255, 255),  # always white text
                thickness=1,
                lineType=cv2.LINE_AA
            )

    # ── show the annotated frame ───────────────────────────────────────────
    cv2.imshow("PPE Live Detection  |  press Q to quit", frame)

    # exit on Q key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── cleanup ───────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("👋 Camera closed. Done.")