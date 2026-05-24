"""
train_face_recognition.py
--------------------------
Treina o sistema usando a biblioteca `face_recognition` (baseada em dlib).
Gera encodings de 128 dimensões para cada face no dataset de treino
e salva em disco. O reconhecimento é feito por distância euclidiana.

Esta abordagem NÃO requer GPU e é muito mais rápida para datasets pequenos.
"""

import os
import json
import pickle
import numpy as np
import face_recognition
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Configurações ──────────────────────────────────────────────────────────────
TRAIN_DIR    = Path("dataset/train")
TEST_DIR     = Path("dataset/test")
MODEL_DIR    = Path("models")
ENCODINGS_FILE = MODEL_DIR / "face_encodings.pkl"
IMG_SIZE     = (160, 160)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR.mkdir(exist_ok=True)


# ── 1. Gera Encodings ─────────────────────────────────────────────────────────
def generate_encodings(data_dir: Path, split_name: str, progress_cb=None):
    """
    Para cada imagem em data_dir/<nome_pessoa>/*.jpg,
    detecta a face e gera o encoding de 128 dimensões.
    """
    encodings, labels = [], []
    failed = 0
    people = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    total_people = len(people)

    print(f"\n[→] Gerando encodings para split '{split_name}' …")
    for i, person_dir in enumerate(tqdm(people, desc=f"  {split_name}")):
        name   = person_dir.name
        images = list(person_dir.glob("*.jpg"))

        if progress_cb:
            progress_cb(i, total_people, name, split_name)

        for img_path in images:
            img = face_recognition.load_image_file(str(img_path))
            locs = face_recognition.face_locations(img, model="hog")

            if not locs:
                failed += 1
                continue

            # Usa apenas a primeira face detectada
            enc = face_recognition.face_encodings(img, known_face_locations=[locs[0]])[0]
            encodings.append(enc)
            labels.append(name)

    print(f"  [✓] {len(encodings)} encodings gerados | {failed} imagens sem face detectada")
    return np.array(encodings), np.array(labels)


# ── 2. Treina Classificadores ─────────────────────────────────────────────────
def train_classifiers(X_train, y_train, X_test, y_test):
    """
    Treina KNN e SVM nos encodings e avalia nos dados de teste.
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    results = {}

    # ── KNN ──────────────────────────────────────────────────────────────────
    print("\n[→] Treinando KNN …")
    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean", n_jobs=-1)
    knn.fit(X_train, y_train_enc)
    knn_acc = knn.score(X_test, y_test_enc)
    print(f"  [✓] KNN Acurácia: {knn_acc*100:.2f}%")

    # ── SVM ──────────────────────────────────────────────────────────────────
    print("[→] Treinando SVM (RBF kernel) …")
    svm = SVC(kernel="rbf", probability=True, C=10, gamma="scale")
    svm.fit(X_train, y_train_enc)
    svm_acc = svm.score(X_test, y_test_enc)
    print(f"  [✓] SVM Acurácia: {svm_acc*100:.2f}%")

    # Relatório completo do melhor modelo
    best_name  = "SVM" if svm_acc >= knn_acc else "KNN"
    best_model = svm   if svm_acc >= knn_acc else knn
    y_pred     = best_model.predict(X_test)

    print(f"\n[✓] Melhor modelo: {best_name} ({max(svm_acc, knn_acc)*100:.2f}%)")
    print("\n" + classification_report(y_test_enc, y_pred,
                                       target_names=le.classes_))

    results = {
        "knn_acc": knn_acc,
        "svm_acc": svm_acc,
        "best":    best_name,
        "label_encoder": le,
        "knn": knn,
        "svm": svm,
    }
    return results


# ── 3. Salva Modelos ──────────────────────────────────────────────────────────
def save_models(X_train, y_train, results):
    data = {
        "encodings": X_train,
        "labels":    y_train,
        "knn":       results["knn"],
        "svm":       results["svm"],
        "label_encoder": results["label_encoder"],
    }
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"\n[✓] Modelos salvos: {ENCODINGS_FILE}")


# ── 4. Plot de Resultados ──────────────────────────────────────────────────────
def plot_results(results):
    fig, ax = plt.subplots(figsize=(7, 4))
    models  = ["KNN (k=3)", "SVM (RBF)"]
    accs    = [results["knn_acc"]*100, results["svm_acc"]*100]
    colors  = ["#4C9BE8", "#E87C4C"]
    bars    = ax.bar(models, accs, color=colors, width=0.4, edgecolor="white")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Comparação KNN vs SVM — face_recognition encodings")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = Path("outputs/comparison_fr.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"[✓] Gráfico salvo: {out}")
    plt.close()


# ── 5. Pipeline & Main ────────────────────────────────────────────────────────
def train_pipeline(progress_cb=None):
    X_train, y_train = generate_encodings(TRAIN_DIR, "train", progress_cb)
    
    # Seletor seguro para split de teste
    X_test, y_test = np.array([]), np.array([])
    if TEST_DIR.exists() and any(d.is_dir() for d in TEST_DIR.iterdir()):
        X_test, y_test = generate_encodings(TEST_DIR, "test", progress_cb)

    print(f"\n[i] X_train: {X_train.shape} | X_test: {X_test.shape}")
    if len(X_train) == 0:
        print("[!] Nenhum encoding de treino foi gerado!")
        return None

    # Fallback se não houver dados de teste
    if len(X_test) == 0:
        X_test, y_test = X_train, y_train

    print(f"[i] Classes: {len(set(y_train))}")

    results = train_classifiers(X_train, y_train, X_test, y_test)
    save_models(X_train, y_train, results)
    plot_results(results)
    return results


def main():
    print("=" * 55)
    print("  Treinamento — face_recognition + dlib (KNN / SVM)")
    print("=" * 55)
    train_pipeline()
    print("\n[✓] Treinamento concluído!")


if __name__ == "__main__":
    main()
