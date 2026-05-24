"""
detect_images.py
----------------
Módulo 1: Detecção e reconhecimento facial em IMAGENS estáticas.

Suporta dois backends:
  --backend tensorflow  → usa modelo MobileNetV2 treinado
  --backend fr          → usa face_recognition (dlib encodings + KNN/SVM)

Uso:
    python src/detect_images.py --input foto.jpg
    python src/detect_images.py --input pasta/
    python src/detect_images.py --input foto.jpg --backend tensorflow
    python src/detect_images.py --input foto.jpg --backend fr --classifier svm
"""

import argparse
import json
import pickle
import cv2
import numpy as np
import face_recognition
from pathlib import Path
import tensorflow as tf

# ── Configurações ──────────────────────────────────────────────────────────────
MODEL_DIR        = Path("models")
TF_MODEL_PATH    = MODEL_DIR / "face_recognition_tf.keras"
FR_ENCODINGS     = MODEL_DIR / "face_encodings.pkl"
CLASS_IDX_PATH   = MODEL_DIR / "class_indices.json"
OUTPUT_DIR       = Path("outputs/detections")
IMG_SIZE         = (160, 160)
CONFIDENCE_THRESH = 0.40   # confiança mínima para exibir nome
BOX_COLOR        = (0, 84, 255)   # azul BGR (igual ao enunciado)
TEXT_BG_COLOR    = (0, 84, 255)
TEXT_COLOR       = (255, 255, 255)
FONT             = cv2.FONT_HERSHEY_SIMPLEX
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# BACKEND: TensorFlow
# ═══════════════════════════════════════════════════════════════
class TFRecognizer:
    def __init__(self):
        print("[→] Carregando modelo TensorFlow …")
        self.model = tf.keras.models.load_model(str(TF_MODEL_PATH))
        with open(CLASS_IDX_PATH) as f:
            self.idx_to_class = json.load(f)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("[✓] Modelo TF carregado!")

    def detect_faces(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        return faces   # lista de (x, y, w, h)

    def recognize(self, img_bgr, x, y, w, h):
        face_crop = img_bgr[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, IMG_SIZE)
        face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB) / 255.0
        face_inp  = np.expand_dims(face_rgb, axis=0)

        preds    = self.model.predict(face_inp, verbose=0)[0]
        class_id = np.argmax(preds)
        conf     = float(preds[class_id])
        name     = self.idx_to_class.get(str(class_id), "desconhecido")
        return name, conf


# ═══════════════════════════════════════════════════════════════
# BACKEND: face_recognition (dlib)
# ═══════════════════════════════════════════════════════════════
class FRRecognizer:
    def __init__(self, classifier="svm"):
        print(f"[→] Carregando encodings face_recognition (classifier={classifier}) …")
        with open(FR_ENCODINGS, "rb") as f:
            data = pickle.load(f)
        self.clf = data[classifier]
        self.le  = data["label_encoder"]
        print("[✓] Encodings carregados!")

    def process(self, img_bgr):
        """Retorna lista de (top, right, bottom, left, name, conf)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(img_rgb, model="hog")
        if not locations:
            return []
        encodings = face_recognition.face_encodings(img_rgb, locations)
        results   = []
        for enc, loc in zip(encodings, locations):
            proba = self.clf.predict_proba([enc])[0]
            idx   = np.argmax(proba)
            conf  = float(proba[idx])
            name  = self.le.inverse_transform([idx])[0]
            results.append((*loc, name, conf))
        return results


# ═══════════════════════════════════════════════════════════════
# Funções de desenho
# ═══════════════════════════════════════════════════════════════
def draw_box(img, x1, y1, x2, y2, name, conf, threshold=CONFIDENCE_THRESH):
    label = f"{name} ({conf:.2f})" if conf >= threshold else f"? ({conf:.2f})"
    color = BOX_COLOR if conf >= threshold else (0, 0, 200)

    # Retângulo principal
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Fundo do texto
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    return img


# ═══════════════════════════════════════════════════════════════
# Processamento de uma imagem
# ═══════════════════════════════════════════════════════════════
def process_image_tf(img_path: Path, recognizer: TFRecognizer, show: bool):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] Não foi possível abrir: {img_path}")
        return

    faces = recognizer.detect_faces(img)
    print(f"  [{img_path.name}] {len(faces)} face(s) detectada(s)")

    for (x, y, w, h) in faces:
        name, conf = recognizer.recognize(img, x, y, w, h)
        draw_box(img, x, y, x+w, y+h, name, conf)

    out_path = OUTPUT_DIR / f"tf_{img_path.name}"
    cv2.imwrite(str(out_path), img)
    print(f"  [✓] Salvo: {out_path}")

    if show:
        cv2.imshow("TensorFlow — Reconhecimento Facial", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_image_fr(img_path: Path, recognizer: FRRecognizer, show: bool):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] Não foi possível abrir: {img_path}")
        return

    detections = recognizer.process(img)
    print(f"  [{img_path.name}] {len(detections)} face(s) detectada(s)")

    for (top, right, bottom, left, name, conf) in detections:
        draw_box(img, left, top, right, bottom, name, conf)

    out_path = OUTPUT_DIR / f"fr_{img_path.name}"
    cv2.imwrite(str(out_path), img)
    print(f"  [✓] Salvo: {out_path}")

    if show:
        cv2.imshow("face_recognition — Reconhecimento Facial", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Detecção facial em imagens estáticas")
    parser.add_argument("--input",      required=True, help="Arquivo de imagem ou pasta")
    parser.add_argument("--backend",    default="fr",  choices=["tensorflow", "fr"],
                        help="Backend de reconhecimento")
    parser.add_argument("--classifier", default="svm", choices=["svm", "knn"],
                        help="Classificador (apenas para backend=fr)")
    parser.add_argument("--show",       action="store_true",
                        help="Exibe janela com resultado")
    args = parser.parse_args()

    print("=" * 55)
    print(f"  Detecção em Imagens — backend: {args.backend}")
    print("=" * 55)

    input_path = Path(args.input)
    if input_path.is_dir():
        images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    else:
        images = [input_path]

    print(f"[i] {len(images)} imagem(ns) encontrada(s)\n")

    if args.backend == "tensorflow":
        rec = TFRecognizer()
        for img in images:
            process_image_tf(img, rec, args.show)
    else:
        rec = FRRecognizer(classifier=args.classifier)
        for img in images:
            process_image_fr(img, rec, args.show)

    print(f"\n[✓] Resultados salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
