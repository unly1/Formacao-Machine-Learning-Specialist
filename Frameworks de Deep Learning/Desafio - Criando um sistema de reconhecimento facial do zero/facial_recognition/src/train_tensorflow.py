"""
train_tensorflow.py
-------------------
Treina um modelo de reconhecimento facial usando TensorFlow/Keras.
Utiliza Transfer Learning com MobileNetV2 pré-treinado no ImageNet
como extrator de features, adicionando camadas densas para classificação.

Fluxo:
    dataset/train/ → Data Augmentation → MobileNetV2 → Dense → Softmax
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, TensorBoard)
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configurações ──────────────────────────────────────────────────────────────
TRAIN_DIR   = Path("dataset/train")
TEST_DIR    = Path("dataset/test")
MODEL_DIR   = Path("models")
IMG_SIZE    = (160, 160)
BATCH_SIZE  = 32
EPOCHS_HEAD = 10     # epochs treinando só a cabeça (base congelada)
EPOCHS_FINE = 20     # epochs de fine-tuning (base descongelada)
LR_HEAD     = 1e-3
LR_FINE     = 1e-5
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR.mkdir(exist_ok=True)


# ── 1. Data Generators ────────────────────────────────────────────────────────
def build_generators():
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    val_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_aug.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    # Salva o mapeamento classe → índice
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open(MODEL_DIR / "class_indices.json", "w") as f:
        json.dump(idx_to_class, f, indent=2)

    return train_gen, val_gen, len(train_gen.class_indices)


# ── 2. Construção do Modelo ────────────────────────────────────────────────────
def build_model(num_classes: int) -> keras.Model:
    """
    MobileNetV2 como base (Transfer Learning) + cabeça de classificação.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # congela na primeira fase

    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(512, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="FaceClassifier_MobileNetV2")
    model.summary()
    return model, base


# ── 3. Callbacks ──────────────────────────────────────────────────────────────
def get_callbacks(phase: str):
    return [
        ModelCheckpoint(
            str(MODEL_DIR / f"best_model_{phase}.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
        TensorBoard(log_dir=f"outputs/logs/{phase}", histogram_freq=1),
    ]


# ── 4. Plot de Histórico ───────────────────────────────────────────────────────
def plot_history(history, title: str, filename: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1.plot(history.history["accuracy"],     label="Treino")
    ax1.plot(history.history["val_accuracy"], label="Validação")
    ax1.set_title("Acurácia"); ax1.set_xlabel("Época")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"],     label="Treino")
    ax2.plot(history.history["val_loss"], label="Validação")
    ax2.set_title("Loss"); ax2.set_xlabel("Época")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path("outputs") / filename
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[✓] Gráfico salvo: {out_path}")
    plt.close()


# ── 5. Avaliação Final ─────────────────────────────────────────────────────────
def evaluate(model, val_gen):
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"\n[✓] Avaliação Final — Loss: {loss:.4f} | Acurácia: {acc*100:.2f}%")
    return acc


# ── 6. Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Treinamento — TensorFlow/Keras (MobileNetV2)")
    print("=" * 55)

    # Verifica GPU
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[i] GPUs disponíveis: {len(gpus)}" + (" (usando GPU)" if gpus else " (usando CPU)"))

    train_gen, val_gen, num_classes = build_generators()
    print(f"[i] Classes encontradas: {num_classes}")

    model, base = build_model(num_classes)

    # ── Fase 1: treina somente a cabeça ──────────────────────────────────────
    print("\n[→] FASE 1 — Treinando cabeça (base congelada) …")
    model.compile(
        optimizer=keras.optimizers.Adam(LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist1 = model.fit(
        train_gen, epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        callbacks=get_callbacks("phase1"),
    )
    plot_history(hist1, "Fase 1 — Cabeça", "history_phase1.png")

    # ── Fase 2: fine-tuning (descongela últimas camadas) ─────────────────────
    print("\n[→] FASE 2 — Fine-tuning (base descongelada) …")
    base.trainable = True
    # Congela as primeiras 100 camadas, treina as últimas
    for layer in base.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(LR_FINE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist2 = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD + EPOCHS_FINE,
        initial_epoch=EPOCHS_HEAD,
        validation_data=val_gen,
        callbacks=get_callbacks("phase2"),
    )
    plot_history(hist2, "Fase 2 — Fine-tuning", "history_phase2.png")

    # ── Salva modelo final ────────────────────────────────────────────────────
    final_path = MODEL_DIR / "face_recognition_tf.keras"
    model.save(str(final_path))
    print(f"\n[✓] Modelo salvo: {final_path}")

    evaluate(model, val_gen)
    print("\n[✓] Treinamento concluído!")


if __name__ == "__main__":
    main()
