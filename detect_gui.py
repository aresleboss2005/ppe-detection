# ==============================================================
# detect_gui.py -- PPE Detection System with Person Tracking
# HOW TO RUN: python detect_gui.py
# REQUIRES:   runs/detect/ppe_model/weights/best.pt
# ==============================================================

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import threading
import datetime
import os
import sys
import glob

try:
    from ultralytics import YOLO
except ImportError:
    import tkinter.messagebox as mb
    r = tk.Tk(); r.withdraw()
    mb.showerror("Missing package", "Run: pip install ultralytics opencv-python pillow")
    sys.exit(1)

# ==============================================================
#  CONFIGURATION
# ==============================================================
# Auto-find latest best.pt
model_dirs = sorted(glob.glob("runs/detect/ppe_model*"))
MODEL_PATH  = "runs/detect/ppe_model/weights/best.pt"
for d in reversed(model_dirs):
    best = os.path.join(d, "weights", "best.pt")
    if os.path.exists(best):
        MODEL_PATH = best
        break

CONFIDENCE   = 0.45
CAMERA_INDEX = 0
CAM_W        = 640
CAM_H        = 480

# How many frames a person stays tracked without being seen
PERSON_TIMEOUT = 30   # frames

# How many frames to wait before alerting same person again
ALERT_COOLDOWN = 90   # frames (~3 sec at 30fps)

MISSING_PPE = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
WEARING_PPE = {"Hardhat", "Mask", "Safety Vest", "Gloves"}
SKIP_CLASSES = {
    "machinery", "vehicle", "Excavator", "Ladder", "Safety Cone",
    "SUV", "bus", "dump truck", "mini-van", "sedan", "semi",
    "trailer", "truck", "truck and trailer", "van",
    "wheel loader", "fire hydrant",
}

# ==============================================================
#  COLORS
# ==============================================================
BG_DARK      = "#0d1117"
BG_CARD      = "#161b22"
BG_PANEL     = "#1c2128"
ACCENT_BLUE  = "#1f6feb"
ACCENT_CYAN  = "#00b4d8"
GREEN        = "#3fb950"
RED          = "#f85149"
ORANGE       = "#d29922"
TEXT_PRIMARY = "#e6edf3"
TEXT_DIM     = "#8b949e"
BORDER       = "#30363d"

# ==============================================================
#  PERSON TRACKER
# ==============================================================
class PersonTracker:
    """
    Tracks persons detected in the frame.
    Assigns each person a unique ID based on bounding box position.
    Fires alert only ONCE per person per non-compliance event.
    """
    def __init__(self):
        self.persons    = {}   # id -> {box, ppe, frames_seen, alerted, timeout}
        self.next_id    = 1
        self.iou_thresh = 0.3  # minimum overlap to match same person

    def _iou(self, a, b):
        """Intersection over Union between two boxes (x1,y1,x2,y2)."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2-ax1) * (ay2-ay1)
        area_b = (bx2-bx1) * (by2-by1)
        return inter / (area_a + area_b - inter)

    def _match(self, box):
        """Find existing person ID that matches this box."""
        best_id  = None
        best_iou = 0.0
        for pid, data in self.persons.items():
            iou = self._iou(box, data["box"])
            if iou > best_iou:
                best_iou = iou
                best_id  = pid
        if best_iou >= self.iou_thresh:
            return best_id
        return None

    def update(self, person_boxes, ppe_detections):
        """
        person_boxes   : list of (x1,y1,x2,y2) for each Person detected
        ppe_detections : list of (class_name, x1,y1,x2,y2) for PPE items

        Returns list of:
        { id, box, hardhat, vest, mask, gloves, non_compliant, new_alert }
        """
        # Age all existing persons
        for pid in list(self.persons.keys()):
            self.persons[pid]["timeout"] -= 1
            if self.persons[pid]["timeout"] <= 0:
                del self.persons[pid]
            else:
                self.persons[pid]["alert_cooldown"] = max(
                    0, self.persons[pid].get("alert_cooldown", 0) - 1)

        # Match or create person entries
        matched_ids = set()
        for box in person_boxes:
            pid = self._match(box)
            if pid is None:
                pid = self.next_id
                self.next_id += 1
                self.persons[pid] = {
                    "box"            : box,
                    "hardhat"        : None,
                    "vest"           : None,
                    "mask"           : None,
                    "gloves"         : None,
                    "timeout"        : PERSON_TIMEOUT,
                    "alert_cooldown" : 0,
                    "alerted"        : False,
                }
            else:
                self.persons[pid]["box"]     = box
                self.persons[pid]["timeout"] = PERSON_TIMEOUT
            matched_ids.add(pid)

            # Reset PPE status for this frame
            self.persons[pid]["hardhat"] = None
            self.persons[pid]["vest"]    = None
            self.persons[pid]["mask"]    = None
            self.persons[pid]["gloves"]  = None

        # Assign PPE detections to nearest person
        for (cls, px1, py1, px2, py2) in ppe_detections:
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            best_pid  = None
            best_dist = float("inf")
            for pid in matched_ids:
                bx1, by1, bx2, by2 = self.persons[pid]["box"]
                bcx = (bx1 + bx2) / 2
                bcy = (by1 + by2) / 2
                dist = abs(pcx - bcx) + abs(pcy - bcy)
                if dist < best_dist:
                    best_dist = dist
                    best_pid  = pid

            if best_pid is None:
                continue

            p = self.persons[best_pid]
            if cls == "Hardhat":
                p["hardhat"] = True
            elif cls == "NO-Hardhat":
                p["hardhat"] = False
            elif cls == "Safety Vest":
                p["vest"] = True
            elif cls == "NO-Safety Vest":
                p["vest"] = False
            elif cls == "Mask":
                p["mask"] = True
            elif cls == "NO-Mask":
                p["mask"] = False
            elif cls == "Gloves":
                p["gloves"] = True

        # Build result list
        results = []
        for pid in matched_ids:
            p = self.persons[pid]
            non_compliant = (
                    p["hardhat"] is False or
                    p["vest"]    is False or
                    p["mask"]    is False
            )
            # Fire alert only once per non-compliance event
            new_alert = False
            if non_compliant and p["alert_cooldown"] == 0:
                new_alert               = True
                p["alert_cooldown"]     = ALERT_COOLDOWN
                p["alerted"]            = True

            results.append({
                "id"            : pid,
                "box"           : p["box"],
                "hardhat"       : p["hardhat"],
                "vest"          : p["vest"],
                "mask"          : p["mask"],
                "gloves"        : p["gloves"],
                "non_compliant" : non_compliant,
                "new_alert"     : new_alert,
            })

        return results

    def reset(self):
        self.persons = {}
        self.next_id = 1


# ==============================================================
#  MAIN APP
# ==============================================================
class PPEApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PPE Detection System")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        self.running      = False
        self.sound_on     = True
        self.cap          = None
        self.model        = None
        self.frame_thread = None
        self.alert_count  = 0
        self.tracker      = PersonTracker()

        self.hardhat_status = tk.StringVar(value="--")
        self.vest_status    = tk.StringVar(value="--")
        self.persons_var    = tk.StringVar(value="0")
        self.alert_var      = tk.StringVar(value="0")
        self.status_var     = tk.StringVar(value="EN ATTENTE DE DETECTION")
        self.online_var     = tk.StringVar(value="OFFLINE")

        # Stores last known PPE state per person to avoid log spam
        self.last_person_states = {}

        self._build_ui()
        self._load_model()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----------------------------------------------------------
    #  MODEL
    # ----------------------------------------------------------
    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            self._log("!  Modele introuvable -- lancez train.py")
            self.btn_start.config(state="disabled")
            return
        try:
            self.model = YOLO(MODEL_PATH)
            self._log(f"OK  Modele charge : {MODEL_PATH}")
        except Exception as e:
            self._log(f"ERREUR: {e}")
            self.btn_start.config(state="disabled")

    # ----------------------------------------------------------
    #  UI
    # ----------------------------------------------------------
    def _build_ui(self):
        # TOP BAR
        topbar = tk.Frame(self.root, bg=BG_CARD, height=48)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="* P P E   D E T E C T I O N   S Y S T E M",
                 bg=BG_CARD, fg=ACCENT_CYAN,
                 font=("Courier", 13, "bold")).pack(side="left", padx=16)

        self.online_label = tk.Label(topbar, textvariable=self.online_var,
                                     bg=BG_CARD, fg=RED,
                                     font=("Courier", 10, "bold"))
        self.online_label.pack(side="right", padx=20)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        # MAIN ROW
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill="both", expand=False, padx=10, pady=10)

        # LEFT: camera
        left = tk.Frame(main, bg=BG_DARK)
        left.pack(side="left", anchor="n")

        self.cam_canvas = tk.Canvas(
            left, width=CAM_W, height=CAM_H,
            bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER
        )
        self.cam_canvas.pack()

        self.cam_text = self.cam_canvas.create_text(
            CAM_W // 2, CAM_H // 2,
            text="Camera inactive\nAppuie sur DEMARRER",
            fill=TEXT_DIM, font=("Courier", 12), justify="center"
        )

        # Status bar
        status_bar = tk.Frame(left, bg=BG_PANEL, height=36,
                              highlightthickness=1, highlightbackground=BORDER)
        status_bar.pack(fill="x", pady=(6, 0))
        status_bar.pack_propagate(False)
        tk.Label(status_bar, textvariable=self.status_var,
                 bg=BG_PANEL, fg=ACCENT_CYAN,
                 font=("Courier", 10, "bold")).pack(expand=True)

        # Buttons
        btn_row = tk.Frame(left, bg=BG_DARK)
        btn_row.pack(fill="x", pady=(6, 0))

        self.btn_start = tk.Button(
            btn_row, text="DEMARRER", font=("Courier", 10, "bold"),
            bg=ACCENT_BLUE, fg="white", relief="flat", cursor="hand2",
            activebackground="#388bfd", pady=9, command=self._toggle_camera
        )
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.btn_sound = tk.Button(
            btn_row, text="SON ON", font=("Courier", 10, "bold"),
            bg=BG_CARD, fg=TEXT_PRIMARY, relief="flat", cursor="hand2",
            activebackground=BG_PANEL, pady=9,
            highlightthickness=1, highlightbackground=BORDER,
            command=self._toggle_sound
        )
        self.btn_sound.pack(side="left", fill="x", expand=True, padx=3)

        tk.Button(
            btn_row, text="VIDER LOGS", font=("Courier", 10, "bold"),
            bg=BG_CARD, fg=TEXT_PRIMARY, relief="flat", cursor="hand2",
            activebackground=BG_PANEL, pady=9,
            highlightthickness=1, highlightbackground=BORDER,
            command=self._clear_logs
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        # RIGHT: stats
        right = tk.Frame(main, bg=BG_DARK, width=300)
        right.pack(side="left", fill="y", padx=(12, 0), anchor="n")
        right.pack_propagate(False)

        # Stat cards
        self._make_card(right, "CASQUE",            self.hardhat_status, TEXT_PRIMARY)
        self._make_card(right, "GILET DE SECURITE", self.vest_status,    TEXT_PRIMARY)
        self._make_card(right, "PERSONNES ACTIVES", self.persons_var,    ACCENT_CYAN)
        self._make_card(right, "ALERTES",           self.alert_var,      RED)

        # Person tracking panel
        tk.Label(right, text="SUIVI PAR PERSONNE",
                 bg=BG_DARK, fg=TEXT_DIM,
                 font=("Courier", 9, "bold")).pack(fill="x", pady=(8, 2))

        # Column headers for person tracking
        hdr = tk.Frame(right, bg=BG_PANEL,
                       highlightthickness=1, highlightbackground=BORDER)
        hdr.pack(fill="x")
        for label, w in [("ID", 4), ("Casque", 8), ("Gilet", 7), ("Masque", 8), ("Statut", 8)]:
            tk.Label(hdr, text=label, bg=BG_PANEL, fg=TEXT_DIM,
                     font=("Courier", 8), width=w, anchor="w",
                     padx=4, pady=3).pack(side="left")

        # Person tracking log
        person_outer = tk.Frame(right, bg=BG_CARD,
                                highlightthickness=1, highlightbackground=BORDER)
        person_outer.pack(fill="x")

        self.person_text = tk.Text(
            person_outer, bg=BG_CARD, fg=TEXT_PRIMARY,
            font=("Courier", 8), relief="flat",
            state="disabled", wrap="none", height=6
        )
        self.person_text.pack(fill="x", padx=4, pady=4)
        self.person_text.tag_config("ok",      foreground=GREEN)
        self.person_text.tag_config("missing", foreground=RED)
        self.person_text.tag_config("unknown", foreground=TEXT_DIM)

        # History log
        tk.Label(right, text="HISTORIQUE DES ALERTES",
                 bg=BG_DARK, fg=TEXT_DIM,
                 font=("Courier", 9, "bold")).pack(fill="x", pady=(8, 2))

        hdr2 = tk.Frame(right, bg=BG_PANEL,
                        highlightthickness=1, highlightbackground=BORDER)
        hdr2.pack(fill="x")
        for label, w in [("Heure", 9), ("Personne", 10), ("EPI manquant", 14)]:
            tk.Label(hdr2, text=label, bg=BG_PANEL, fg=TEXT_DIM,
                     font=("Courier", 8), width=w, anchor="w",
                     padx=4, pady=3).pack(side="left")

        log_outer = tk.Frame(right, bg=BG_CARD,
                             highlightthickness=1, highlightbackground=BORDER)
        log_outer.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            log_outer, bg=BG_CARD, fg=TEXT_PRIMARY,
            font=("Courier", 8), relief="flat",
            state="disabled", wrap="none"
        )
        sb = tk.Scrollbar(log_outer, command=self.log_text.yview,
                          bg=BG_PANEL, troughcolor=BG_CARD)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_text.tag_config("alert", foreground=RED)
        self.log_text.tag_config("info",  foreground=TEXT_DIM)

    def _make_card(self, parent, title, var, val_color):
        card = tk.Frame(parent, bg=BG_CARD, padx=12, pady=8,
                        highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="x", pady=(0, 5))
        tk.Label(card, text=title, bg=BG_CARD, fg=TEXT_DIM,
                 font=("Courier", 9, "bold"), anchor="w").pack(fill="x")
        tk.Label(card, textvariable=var, bg=BG_CARD, fg=val_color,
                 font=("Courier", 14, "bold"), anchor="w").pack(fill="x")

    # ----------------------------------------------------------
    #  CAMERA
    # ----------------------------------------------------------
    def _toggle_camera(self):
        if self.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self._log("ERREUR: impossible d'ouvrir la camera")
            return
        self.running = True
        self.tracker.reset()
        self.btn_start.config(text="ARRETER", bg=RED, activebackground="#da3633")
        self.online_var.set("ONLINE")
        self.online_label.config(fg=GREEN)
        self.status_var.set("DETECTION EN COURS...")
        self.cam_canvas.itemconfig(self.cam_text, text="")
        self.frame_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.frame_thread.start()

    def _stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.tracker.reset()
        self.last_person_states = {}
        self.cam_canvas.delete("frame")
        self.cam_canvas.itemconfig(self.cam_text,
                                   text="Camera inactive\nAppuie sur DEMARRER")
        self.btn_start.config(text="DEMARRER", bg=ACCENT_BLUE,
                              activebackground="#388bfd")
        self.online_var.set("OFFLINE")
        self.online_label.config(fg=RED)
        self.status_var.set("EN ATTENTE DE DETECTION")
        self.hardhat_status.set("--")
        self.vest_status.set("--")
        self.persons_var.set("0")

    # ----------------------------------------------------------
    #  DETECTION LOOP
    # ----------------------------------------------------------
    def _detection_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            person_boxes    = []
            ppe_detections  = []
            hardhat_ok      = hardhat_missing = False
            vest_ok         = vest_missing    = False

            if self.model:
                results = self.model.track(source=frame, conf=CONFIDENCE, persist=True, verbose=False)

                for result in results:
                    for box in result.boxes:
                        class_id   = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])

                        if class_name in SKIP_CLASSES:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if class_name == "Person":
                            person_boxes.append((x1, y1, x2, y2))
                            color = (34, 165, 210)
                        elif class_name in MISSING_PPE:
                            ppe_detections.append((class_name, x1, y1, x2, y2))
                            color = (80, 40, 248)
                            if "Hardhat" in class_name: hardhat_missing = True
                            if "Vest"    in class_name: vest_missing    = True
                        elif class_name in WEARING_PPE:
                            ppe_detections.append((class_name, x1, y1, x2, y2))
                            color = (63, 185, 80)
                            if class_name == "Hardhat":     hardhat_ok = True
                            if "Vest" in class_name:        vest_ok    = True
                        else:
                            color = (150, 150, 150)

                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} {confidence:.0%}"
                        (tw, th), bl = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        ly = max(y1 - th - bl - 4, 0)
                        cv2.rectangle(frame, (x1, ly), (x1 + tw + 6, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 3, y1 - bl - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)

            # Update person tracker
            tracked = self.tracker.update(person_boxes, ppe_detections)

            # Draw person ID on frame
            for p in tracked:
                bx1, by1, bx2, by2 = p["box"]
                id_label = f"P{p['id']}"
                cv2.putText(frame, id_label, (bx1, by2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 200, 255), 2, cv2.LINE_AA)

            # Check if any alert fired this frame
            new_alerts = [p for p in tracked if p["new_alert"]]

            # Update UI
            self.hardhat_status.set("ABSENT" if hardhat_missing else ("PORTE" if hardhat_ok else "--"))
            self.vest_status.set("ABSENT"    if vest_missing    else ("PORTE" if vest_ok    else "--"))
            self.persons_var.set(str(len(tracked)))

            if new_alerts:
                self.alert_count += len(new_alerts)
                self.alert_var.set(str(self.alert_count))
                self.status_var.set("ALERTE -- EPI MANQUANT !")
                for p in new_alerts:
                    missing = []
                    if p["hardhat"] is False: missing.append("Casque")
                    if p["vest"]    is False: missing.append("Gilet")
                    if p["mask"]    is False: missing.append("Masque")
                    self.root.after(0, self._log_alert, p["id"], missing)
                if self.sound_on:
                    try:
                        import winsound
                        winsound.Beep(1000, 300)
                    except Exception:
                        pass
            else:
                if tracked:
                    self.status_var.set("DETECTION EN COURS...")
                else:
                    self.status_var.set("EN ATTENTE DE DETECTION")

            # Clean up states for persons no longer tracked
            active_ids = {p["id"] for p in tracked}
            for pid in list(self.last_person_states.keys()):
                if pid not in active_ids:
                    del self.last_person_states[pid]

            # Update person tracking panel
            self.root.after(0, self._update_person_panel, tracked)

            # Display frame
            h, w = frame.shape[:2]
            scale = min(CAM_W / w, CAM_H / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh))
            pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            self.root.after(0, self._update_canvas, pil_img)

        self.root.after(0, self._stop_camera)

    def _update_canvas(self, pil_img):
        canvas_img = Image.new("RGB", (CAM_W, CAM_H), (13, 17, 23))
        ox = (CAM_W - pil_img.width)  // 2
        oy = (CAM_H - pil_img.height) // 2
        canvas_img.paste(pil_img, (ox, oy))
        imgtk = ImageTk.PhotoImage(image=canvas_img)
        self.cam_canvas.delete("frame")
        self.cam_canvas.create_image(0, 0, anchor="nw", image=imgtk, tags="frame")
        self.cam_canvas._imgtk = imgtk

    # ----------------------------------------------------------
    #  PERSON TRACKING PANEL
    # ----------------------------------------------------------
    def _update_person_panel(self, tracked):
        self.person_text.config(state="normal")
        self.person_text.delete("1.0", "end")

        if not tracked:
            self.person_text.insert("end", "  Aucune personne detectee\n", "unknown")
        else:
            for p in tracked:
                pid = f"P{p['id']:<3}"

                # Hardhat
                if p["hardhat"] is True:
                    h_str = "OK    "
                    h_tag = "ok"
                elif p["hardhat"] is False:
                    h_str = "ABSENT"
                    h_tag = "missing"
                else:
                    h_str = "?     "
                    h_tag = "unknown"

                # Vest
                if p["vest"] is True:
                    v_str = "OK    "
                    v_tag = "ok"
                elif p["vest"] is False:
                    v_str = "ABSENT"
                    v_tag = "missing"
                else:
                    v_str = "?     "
                    v_tag = "unknown"

                # Mask
                if p["mask"] is True:
                    m_str = "OK    "
                    m_tag = "ok"
                elif p["mask"] is False:
                    m_str = "ABSENT"
                    m_tag = "missing"
                else:
                    m_str = "?     "
                    m_tag = "unknown"

                # Overall status
                if p["non_compliant"]:
                    s_str = "ALERTE"
                    s_tag = "missing"
                else:
                    s_str = "OK    "
                    s_tag = "ok"

                # Write line with color tags per segment
                self.person_text.insert("end", f"  {pid} ", "unknown")
                self.person_text.insert("end", f"{h_str}  ", h_tag)
                self.person_text.insert("end", f"{v_str}  ", v_tag)
                self.person_text.insert("end", f"{m_str}  ", m_tag)
                self.person_text.insert("end", f"{s_str}\n", s_tag)

        self.person_text.config(state="disabled")

    # ----------------------------------------------------------
    #  LOGGING
    # ----------------------------------------------------------
    def _log_alert(self, person_id, missing_items):
        now     = datetime.datetime.now().strftime("%H:%M:%S")
        missing = ", ".join(missing_items) if missing_items else "Inconnu"
        line    = f"{now:<10}P{person_id:<9}{missing}\n"
        self.log_text.config(state="normal")
        self.log_text.insert("end", line, "alert")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _log(self, message):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"{now:<10}{message}\n", "info")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_logs(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self.person_text.config(state="normal")
        self.person_text.delete("1.0", "end")
        self.person_text.config(state="disabled")
        self.alert_count = 0
        self.alert_var.set("0")
        self.tracker.reset()
        self.last_person_states = {}

    # ----------------------------------------------------------
    #  SOUND / CLOSE
    # ----------------------------------------------------------
    def _toggle_sound(self):
        self.sound_on = not self.sound_on
        self.btn_sound.config(text="SON ON" if self.sound_on else "SON OFF")

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ==============================================================
#  ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = PPEApp(root)
    root.mainloop()