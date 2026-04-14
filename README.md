<div align="center">

```
██████╗ ██████╗ ███████╗    ██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗ 
██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
██████╔╝██████╔╝█████╗      ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝
██╔═══╝ ██╔═══╝ ██╔══╝      ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
██║     ██║     ███████╗    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
╚═╝     ╚═╝     ╚══════╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

# PPE Detection System
n'est pas à 100% fonctionnel

**Système de détection d'équipements de protection individuelle en temps réel**  
*Powered by YOLOv8 · Python · OpenCV · Tkinter*

---

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=for-the-badge)

</div>

---

## 📋 Table des matières

- [À propos du projet](#-à-propos-du-projet)
- [Fonctionnalités](#-fonctionnalités)
- [Classes détectées](#-classes-détectées)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Interface graphique](#-interface-graphique)
- [Entraînement du modèle](#-entraînement-du-modèle)
- [Créer un exécutable](#-créer-un-exécutable)
- [Résolution des erreurs](#-résolution-des-erreurs)

---

## 🎯 À propos du projet

Ce projet est un **système de détection automatique d'EPI (Équipements de Protection Individuelle)** sur les chantiers de construction. Il utilise le modèle de deep learning **YOLOv8** pour analyser en temps réel un flux vidéo (webcam ou caméra IP) et détecter si les travailleurs portent correctement leurs équipements de sécurité.

### Contexte
- Dataset : **Construction Site Safety** (Roboflow Universe — 2801 images, 25 classes)
- Modèle : **YOLOv8n** fine-tuné sur le dataset
- Interface : **Tkinter** dark theme

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 🎥 **Détection live** | Analyse en temps réel via webcam ou caméra IP |
| 🟢 **EPI porté** | Boîte verte quand l'équipement est correctement porté |
| 🔴 **EPI manquant** | Boîte rouge + alerte sonore quand l'équipement est absent |
| 🟠 **Personne détectée** | Boîte orange pour les personnes sans info EPI |
| 📊 **Statistiques live** | Compteur de détections et d'alertes en temps réel |
| 📋 **Historique** | Log horodaté de toutes les détections |
| 🔔 **Son** | Bip automatique lors d'une alerte (désactivable) |
| 🗑️ **Vider logs** | Remise à zéro des compteurs et de l'historique |

---

## 🏷️ Classes détectées

Le modèle détecte **7 classes** au total :

### ✅ EPI correctement porté
```
Hardhat · Mask · Safety Vest · Gloves
```

### ❌ EPI manquant (déclenche une alerte)
```
NO-Hardhat · NO-Mask · NO-Safety Vest
```
---
## 📁 Structure du projet


```
ppe/
│
├── 📄 data.yaml              ← Configuration du dataset (Roboflow)
├── 📄 requirements.txt       ← Dépendances Python
├── 🐍 train.py               ← Entraînement du modèle YOLOv8
├── 🐍 detect_gui.py          ← Interface graphique de détection
├── ⚙️  build_exe.bat          ← Script pour créer un .exe portable
├── 📄 README.md              ← Ce fichier
│
├── 📁 dataset/
│   ├── train/
│   │   ├── images/           ← Images d'entraînement
│   │   └── labels/           ← Annotations YOLO
│   ├── valid/
│   │   ├── images/           ← Images de validation
│   │   └── labels/
│   └── test/
│       ├── images/           ← Images de test
│       └── labels/
│
└── 📁 runs/                  ← Créé automatiquement après train.py
    └── detect/
        └── ppe_model/
            └── weights/
                ├── best.pt   ← Meilleur modèle ✅
                └── last.pt   ← Dernier checkpoint
```

---

## ⚙️ Installation

### Prérequis
- Python **3.8 → 3.11**
- Windows 10/11
- Webcam (ou fichier vidéo)

### Étape 1 — Cloner le projet
```bash
git clone https://github.com/aresleboss2005/Ppe_detector.git
cd Ppe_detector
```

### Étape 2 — Créer un environnement virtuel
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Étape 3 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 4 — Installer Visual C++ Runtime (Windows)
Télécharger et installer : https://aka.ms/vs/17/release/vc_redist.x64.exe

---

## 🚀 Utilisation

### Étape 1 — Entraîner le modèle
```bash
python train.py
```
> ⏱️ Durée : ~30 min sur CPU, ~10 min sur GPU  
> 📁 Résultat : `runs/detect/ppe_model/weights/best.pt`

### Étape 2 — Lancer l'interface
```bash
python detect_gui.py
```

### Utiliser une vidéo à la place de la webcam
Dans `detect_gui.py`, modifier :
```python
# Webcam (défaut)
CAMERA_INDEX = 0

# Fichier vidéo local
CAMERA_INDEX = r"C:\chemin\vers\video.mp4"

# Caméra IP (chantier)
CAMERA_INDEX = "rtsp://admin:password@192.168.1.100:554/stream"
```

---

## 🖥️ Interface graphique

```
┌─ PPE DETECTION SYSTEM ──────────────────────────── ONLINE ─┐
│                                                              │
│  ┌─────────────────────────────┐  ┌──────────────────────┐  │
│  │                             │  │ 🪖 CASQUE            │  │
│  │                             │  │    PORTE ✅          │  │
│  │      FLUX CAMERA LIVE       │  ├──────────────────────┤  │
│  │                             │  │ 🦺 GILET DE SECURITE │  │
│  │  [vert] Hardhat OK 94%      │  │    ABSENT ❌         │  │
│  │  [rouge] NO-Vest MISSING    │  ├──────────────────────┤  │
│  │                             │  │ 📊 DETECTIONS : 42   │  │
│  └─────────────────────────────┘  ├──────────────────────┤  │
│                                   │ 🚨 ALERTES : 3       │  │
│  ⚠ ALERTE — EPI MANQUANT !       ├──────────────────────┤  │
│                                   │ HISTORIQUE           │  │
│  [DEMARRER] [SON ON] [VIDER LOGS] │ 14:32:01 NO-Hardhat  │  │
└───────────────────────────────────┴──────────────────────┘  │
```

| Bouton | Action |
|---|---|
| **DEMARRER** | Lance la caméra et la détection |
| **ARRETER** | Stoppe la caméra |
| **SON ON/OFF** | Active ou désactive le bip d'alerte |
| **VIDER LOGS** | Remet les compteurs à zéro |

---

## 🧠 Entraînement du modèle

Le fichier `train.py` utilise **YOLOv8n** (nano) fine-tuné sur le dataset Construction Site Safety.

### Paramètres d'entraînement
```python
epochs   = 20      # nombre de passes sur le dataset
imgsz    = 640     # résolution des images
batch    = 8       # images par batch (réduire à 4 si erreur mémoire)
patience = 10      # arrêt automatique si pas d'amélioration
device   = 0       # GPU (changer en 'cpu' si pas de GPU)
```

### Comprendre les métriques
| Métrique | Signification | Objectif |
|---|---|---|
| `box_loss` | Précision de localisation des boîtes | Diminuer |
| `cls_loss` | Précision de classification des classes | Diminuer |
| `dfl_loss` | Précision des bords des boîtes | Diminuer |

### Continuer un entraînement existant
```python
# Dans train.py — repart du meilleur modèle précédent
model = YOLO("runs/detect/ppe_model/weights/best.pt")
```

---

## 📦 Créer un exécutable

Pour partager l'application sans installer Python :

```bash
build_exe.bat
```

L'exécutable sera dans `dist/PPE_Detection/PPE_Detection.exe`.  
Copier le dossier `dist/PPE_Detection/` entier sur n'importe quel PC Windows.

---

## 🔧 Résolution des erreurs

### DLL initialization failed (torch)
```
OSError: [WinError 1114] A dynamic link library failed
```
**Solution :** Installer Visual C++ Runtime :  
👉 https://aka.ms/vs/17/release/vc_redist.x64.exe

### Caméra introuvable
```
ERREUR: impossible d'ouvrir la camera
```
**Solution :** Changer `CAMERA_INDEX = 0` en `1` ou `2` dans `detect_gui.py`

### Modèle introuvable
```
Modele introuvable -- lancez train.py d'abord
```
**Solution :** Lancer `python train.py` avant `detect_gui.py`

### Out of memory pendant l'entraînement
**Solution :** Réduire `batch=8` à `batch=4` dans `train.py`

---

## 📊 Dataset

| Info | Valeur |
|---|---|
| Source | Roboflow Universe — Construction Site Safety v28 |
| Images totales | 2801 |
| Images train | 2605 |
| Images validation | 114 |
| Images test | 82 |
| Résolution | 640×640 |
| Format | YOLOv8 |
| License | CC BY 4.0 |

---

## 👨‍💻 Auteur

**Ayman** — Projet de fin de module  
Système de détection EPI sur chantier de construction  

---

<div align="center">

*Fait avec ❤️ et beaucoup de epochs*

</div>
