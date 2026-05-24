"""
detect_webcam.py
----------------
Módulo 2: Detecção e reconhecimento facial em TEMPO REAL via webcam.

Suporta os mesmos dois backends do detect_images.py.

Controles durante a execução:
  Q / ESC → sair
  S       → salvar screenshot do frame atual
  B       → alternar entre backends (TF ↔ face_recognition)
  F       → alternar FPS display

Uso:
    python src/detect_webcam.py
    python src/detect_webcam.py --backend tensorflow
    python src/detect_webcam.py --backend fr --classifier knn --camera 0
"""

import argparse
import json
import pickle
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import face_recognition
import tensorflow as tf

# ── Configurações ──────────────────────────────────────────────────────────────
MODEL_DIR         = Path("models")
TF_MODEL_PATH     = MODEL_DIR / "face_recognition_tf.keras"
FR_ENCODINGS      = MODEL_DIR / "face_encodings.pkl"
CLASS_IDX_PATH    = MODEL_DIR / "class_indices.json"
SCREENSHOTS_DIR   = Path("outputs/screenshots")
IMG_SIZE          = (160, 160)
CONFIDENCE_THRESH = 0.40
FRAME_SKIP        = 2       # processa reconhecimento a cada N frames (performance)
BOX_COLOR         = (0, 84, 255)
TEXT_COLOR        = (255, 255, 255)
FONT              = cv2.FONT_HERSHEY_SIMPLEX
# ──────────────────────────────────────────────────────────────────────────────

SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# BACKEND: TensorFlow (Haar Cascade + MobileNetV2)
# ═══════════════════════════════════════════════════════════════
class TFRecognizer:
    def __init__(self):
        print("[->] Carregando modelo TensorFlow ...")
        self.model = tf.keras.models.load_model(str(TF_MODEL_PATH))
        with open(CLASS_IDX_PATH) as f:
            self.idx_to_class = json.load(f)
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("[OK] TF pronto!")

    def process_frame(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        results = []
        for (x, y, w, h) in faces:
            crop = frame[y:y+h, x:x+w]
            crop = cv2.resize(crop, IMG_SIZE)
            inp  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) / 255.0
            inp  = np.expand_dims(inp, axis=0)
            pred = self.model.predict(inp, verbose=0)[0]
            cid  = np.argmax(pred)
            conf = float(pred[cid])
            name = self.idx_to_class.get(str(cid), "desconhecido")
            results.append((x, y, x+w, y+h, name, conf))
        return results


# ═══════════════════════════════════════════════════════════════
# BACKEND: face_recognition (dlib HOG + SVM/KNN)
# ═══════════════════════════════════════════════════════════════
class FRRecognizer:
    def __init__(self, classifier="svm"):
        print(f"[->] Carregando face_recognition ({classifier}) ...")
        with open(FR_ENCODINGS, "rb") as f:
            data = pickle.load(f)
        self.clf = data[classifier]
        self.le  = data["label_encoder"]
        print("[OK] face_recognition pronto!")

    def process_frame(self, frame):
        # Reduz resolução para acelerar detecção
        small  = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs   = face_recognition.face_locations(rgb, model="hog")
        encs   = face_recognition.face_encodings(rgb, locs)

        results = []
        for enc, (top, right, bottom, left) in zip(encs, locs):
            # Escala de volta para resolução original
            top, right, bottom, left = top*2, right*2, bottom*2, left*2
            proba = self.clf.predict_proba([enc])[0]
            idx   = np.argmax(proba)
            conf  = float(proba[idx])
            name  = self.le.inverse_transform([idx])[0]
            results.append((left, top, right, bottom, name, conf))
        return results


# ═══════════════════════════════════════════════════════════════
# Funções de desenho (HUD e boxes)
# ═══════════════════════════════════════════════════════════════
def draw_face_box(frame, x1, y1, x2, y2, name, conf):
    color  = BOX_COLOR if conf >= CONFIDENCE_THRESH else (50, 50, 200)
    label  = f"{name} ({conf:.2f})" if conf >= CONFIDENCE_THRESH else f"? ({conf:.2f})"

    # Cantos decorativos (estilo "tech")
    corner = 15
    thick  = 2
    for (cx, cy, dx, dy) in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx*corner, cy), color, thick)
        cv2.line(frame, (cx, cy), (cx, cy + dy*corner), color, thick)

    # Retângulo translúcido de fundo
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    # Etiqueta
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_hud(frame, fps, backend, n_faces, show_fps):
    h, w = frame.shape[:2]

    # Fundo do HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 35), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    info = (f"[{backend.upper()}]  Faces: {n_faces}"
            + (f"  |  FPS: {fps:.1f}" if show_fps else ""))
    cv2.putText(frame, info, (10, 24), FONT, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # Legenda de teclas
    keys = "Q=Sair  S=Screenshot  B=Trocar Backend  F=FPS"
    cv2.putText(frame, keys, (10, h - 10), FONT, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# Main — loop de captura
# ═══════════════════════════════════════════════════════════════
def run_webcam(args):
    # Inicializa o backend escolhido
    if args.backend == "tensorflow":
        recognizer = TFRecognizer()
        backend    = "tensorflow"
    else:
        recognizer = FRRecognizer(classifier=args.classifier)
        backend    = f"fr-{args.classifier}"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[!] Não foi possível abrir a câmera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n[OK] Webcam iniciada!")
    print("    Q / ESC -> sair")
    print("    S       -> screenshot")
    print("    B       -> trocar backend")
    print("    F       -> toggle FPS\n")

    frame_count = 0
    last_results = []
    fps_values   = []
    show_fps     = True
    t_prev       = time.time()
    backends     = ["fr-svm", "fr-knn", "tensorflow"]
    backend_idx  = backends.index(backend) if backend in backends else 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Frame não recebido — encerrando.")
            break

        frame_count += 1

        # Reconhecimento a cada FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            last_results = recognizer.process_frame(frame)

        # Desenha os boxes do último reconhecimento
        for (x1, y1, x2, y2, name, conf) in last_results:
            draw_face_box(frame, x1, y1, x2, y2, name, conf)

        # FPS
        t_now = time.time()
        fps_values.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        if len(fps_values) > 30:
            fps_values.pop(0)
        fps = np.mean(fps_values)

        draw_hud(frame, fps, backend, len(last_results), show_fps)
        cv2.imshow("Sistema de Reconhecimento Facial — Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        # Sair
        if key in (ord("q"), ord("Q"), 27):
            break

        # Screenshot
        elif key in (ord("s"), ord("S")):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = SCREENSHOTS_DIR / f"screenshot_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[OK] Screenshot salvo: {path}")

        # Trocar backend
        elif key in (ord("b"), ord("B")):
            backend_idx = (backend_idx + 1) % len(backends)
            next_b      = backends[backend_idx]
            print(f"[->] Trocando para backend: {next_b}")
            if next_b == "tensorflow":
                recognizer = TFRecognizer()
                backend    = "tensorflow"
            else:
                clf = next_b.split("-")[1]
                recognizer = FRRecognizer(classifier=clf)
                backend    = next_b

        # Toggle FPS
        elif key in (ord("f"), ord("F")):
            show_fps = not show_fps

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Webcam encerrada.")


def main():
    parser = argparse.ArgumentParser(description="Reconhecimento facial em tempo real via webcam")
    parser.add_argument("--backend",    default="fr",   choices=["tensorflow", "fr"])
    parser.add_argument("--classifier", default="svm",  choices=["svm", "knn"])
    parser.add_argument("--camera",     default=0,      type=int, help="Índice da câmera")
    args = parser.parse_args()

    print("=" * 55)
    print(f"  Webcam — backend: {args.backend}")
    print("=" * 55)
    run_webcam(args)


if __name__ == "__main__":
    main()
