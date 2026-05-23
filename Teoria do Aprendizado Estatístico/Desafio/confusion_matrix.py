# -*- coding: utf-8 -*-
"""
==============================================================
  CÁLCULO DE MÉTRICAS DE AVALIAÇÃO DE APRENDIZADO
  Projeto: Matriz de Confusão + Métricas de Classificação
  Formação Machine Learning Specialist
==============================================================

Métricas implementadas (Tabela 1 do desafio):
  - Sensibilidade  : VP / (VP + FN)
  - Especificidade : VN / (FP + VN)
  - Acurácia       : (VP + VN) / N
  - Precisão       : VP / (VP + FP)
  - F-score        : 2 x (P x S) / (P + S)

Onde:
  VP = Verdadeiro Positivo  (acertou que É da classe)
  VN = Verdadeiro Negativo  (acertou que NÃO É da classe)
  FP = Falso Positivo       (disse que é, mas não é)
  FN = Falso Negativo       (disse que não é, mas é)
  N  = Total de elementos
"""

# !pip install -q tensorflow

import io
import shutil
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

print(f"TensorFlow Version: {tf.__version__}")

# ══════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL: CÁLCULO DAS MÉTRICAS DO DESAFIO
# Extrai VP, VN, FP, FN de cada classe e calcula todas as métricas
# ══════════════════════════════════════════════════════════════

def calcular_metricas_por_classe(con_mat, classes):
    """
    Para classificação multiclasse, cada classe é tratada como
    um problema binário (One-vs-Rest):

      VP (classe i) = con_mat[i, i]
      FN (classe i) = soma da linha i   - VP
      FP (classe i) = soma da coluna i  - VP
      VN (classe i) = total - VP - FP - FN

    Retorna um DataFrame com todas as métricas por classe
    e os valores médios (macro average).
    """
    total_geral = con_mat.sum()
    resultados = []

    for i, classe in enumerate(classes):
        VP = con_mat[i, i]
        FN = con_mat[i, :].sum() - VP      # linha i  - diagonal
        FP = con_mat[:, i].sum() - VP      # coluna i - diagonal
        VN = total_geral - VP - FN - FP    # resto

        N = total_geral  # total de elementos

        # ── Fórmulas da Tabela 1 do desafio ──
        sensibilidade  = VP / (VP + FN)          if (VP + FN) > 0 else 0.0
        especificidade = VN / (FP + VN)          if (FP + VN) > 0 else 0.0
        acuracia       = (VP + VN) / N           if N > 0         else 0.0
        precisao       = VP / (VP + FP)          if (VP + FP) > 0 else 0.0

        P, S = precisao, sensibilidade
        fscore = 2 * (P * S) / (P + S)          if (P + S) > 0   else 0.0

        resultados.append({
            "Classe":          classe,
            "VP":              int(VP),
            "VN":              int(VN),
            "FP":              int(FP),
            "FN":              int(FN),
            "Sensibilidade":   round(sensibilidade,  4),
            "Especificidade":  round(especificidade, 4),
            "Acuracia":        round(acuracia,        4),
            "Precisao":        round(precisao,        4),
            "F-score":         round(fscore,          4),
        })

    df = pd.DataFrame(resultados)

    # Linha de média macro (média simples entre todas as classes)
    media = {
        "Classe":         "MÉDIA MACRO",
        "VP":             int(df["VP"].sum()),
        "VN":             "-",
        "FP":             int(df["FP"].sum()),
        "FN":             int(df["FN"].sum()),
        "Sensibilidade":  round(df["Sensibilidade"].mean(),  4),
        "Especificidade": round(df["Especificidade"].mean(), 4),
        "Acuracia":       round(df["Acuracia"].mean(),        4),
        "Precisao":       round(df["Precisao"].mean(),        4),
        "F-score":        round(df["F-score"].mean(),         4),
    }
    df = pd.concat([df, pd.DataFrame([media])], ignore_index=True)
    return df


def plotar_metricas(df_metricas, nome_modelo, pasta):
    """
    Gera gráfico de barras com as 5 métricas por classe.
    """
    df_plot = df_metricas[df_metricas["Classe"] != "MÉDIA MACRO"].copy()
    df_plot["Classe"] = df_plot["Classe"].astype(str)

    metricas = ["Sensibilidade", "Especificidade", "Acuracia", "Precisao", "F-score"]
    cores    = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(f"{nome_modelo} — Métricas por Classe", fontsize=14, fontweight="bold")

    for ax, metrica, cor in zip(axes, metricas, cores):
        ax.bar(df_plot["Classe"], df_plot[metrica], color=cor, alpha=0.85, edgecolor="white")
        ax.set_title(metrica, fontweight="bold")
        ax.set_xlabel("Classe")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.axhline(y=df_metricas[df_metricas["Classe"] == "MÉDIA MACRO"][metrica].values[0],
                   color="black", linestyle="--", linewidth=1, label="Média")
        ax.legend(fontsize=8)
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    caminho = os.path.join(pasta, f"metricas_{nome_modelo.replace(' ', '_')}.png")
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return caminho


def escrever_metricas_txt(f, df_metricas, nome_modelo, con_mat_raw):
    """
    Escreve no arquivo .txt a seção completa de métricas de um modelo.
    """
    sep = "=" * 65

    f.write(f"\n{sep}\n")
    f.write(f"  {nome_modelo}: EXTRAÇÃO DE VP, VN, FP, FN\n")
    f.write(f"  (One-vs-Rest — para cada classe da Tabela 1)\n")
    f.write(f"{sep}\n\n")

    f.write("  Legenda:\n")
    f.write("    VP = Verdadeiro Positivo  → acertou que É da classe\n")
    f.write("    VN = Verdadeiro Negativo  → acertou que NÃO É da classe\n")
    f.write("    FP = Falso Positivo       → classificou como classe, mas não é\n")
    f.write("    FN = Falso Negativo       → era da classe, mas não reconheceu\n\n")

    # Tabela VP/VN/FP/FN
    f.write(f"  {'Classe':<12} {'VP':>7} {'VN':>8} {'FP':>7} {'FN':>7}\n")
    f.write("  " + "-" * 44 + "\n")
    for _, row in df_metricas.iterrows():
        f.write(f"  {str(row['Classe']):<12} {str(row['VP']):>7} "
                f"{str(row['VN']):>8} {str(row['FP']):>7} {str(row['FN']):>7}\n")

    f.write(f"\n{sep}\n")
    f.write(f"  {nome_modelo}: MÉTRICAS (Tabela 1 do Desafio)\n")
    f.write(f"{sep}\n\n")

    f.write("  Fórmulas aplicadas:\n")
    f.write("    Sensibilidade  = VP / (VP + FN)\n")
    f.write("    Especificidade = VN / (FP + VN)\n")
    f.write("    Acurácia       = (VP + VN) / N\n")
    f.write("    Precisão       = VP / (VP + FP)\n")
    f.write("    F-score        = 2 × (Precisão × Sensibilidade) / (Precisão + Sensibilidade)\n\n")

    # Tabela de métricas
    header = f"  {'Classe':<12} {'Sensib.':>10} {'Especif.':>10} {'Acurácia':>10} {'Precisão':>10} {'F-score':>10}"
    f.write(header + "\n")
    f.write("  " + "-" * 65 + "\n")

    for _, row in df_metricas.iterrows():
        destaque = "★" if row["Classe"] == "MÉDIA MACRO" else " "
        f.write(f"{destaque} {str(row['Classe']):<12} "
                f"{str(row['Sensibilidade']):>10} "
                f"{str(row['Especificidade']):>10} "
                f"{str(row['Acuracia']):>10} "
                f"{str(row['Precisao']):>10} "
                f"{str(row['F-score']):>10}\n")
        if row["Classe"] == 9:
            f.write("  " + "-" * 65 + "\n")

    f.write("\n")


# ══════════════════════════════════════════════════════════════
# SETUP — DIRETÓRIOS E DADOS
# ══════════════════════════════════════════════════════════════

# Pasta base = diretório onde este script está salvo (sempre dentro de Desafio/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logdir_base = os.path.join(BASE_DIR, 'log')
os.makedirs(logdir_base, exist_ok=True)
if os.name == 'nt':
    import ctypes
    buf_size = 256
    buffer = ctypes.create_unicode_buffer(buf_size)
    ctypes.windll.kernel32.GetShortPathNameW(logdir_base, buffer, buf_size)
    logdir = buffer.value
else:
    logdir = logdir_base

shutil.rmtree(logdir, ignore_errors=True)
os.makedirs(logdir, exist_ok=True)

RESULTADOS_DIR = os.path.join(BASE_DIR, 'resultados')
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Carrega MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images  = test_images.reshape((10000, 28, 28, 1))  / 255.0

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# ══════════════════════════════════════════════════════════════
# MODELO 1 — CNN base (sem callback de CM)
# ══════════════════════════════════════════════════════════════

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=train_images,
                    y=train_labels,
                    epochs=5,
                    validation_data=(test_images, test_labels))

# Predições e matriz de confusão — Modelo 1
y_true = test_labels
y_pred = np.argmax(model.predict(test_images), axis=1)

con_mat1_raw  = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
con_mat1_norm = np.around(con_mat1_raw.astype('float') /
                          con_mat1_raw.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat1_df   = pd.DataFrame(con_mat1_norm, index=classes, columns=classes)

# ── Calcula métricas do desafio para Modelo 1 ──
df_metricas1 = calcular_metricas_por_classe(con_mat1_raw, classes)

# Heatmap Modelo 1
figure1 = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat1_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Modelo 1 — Matriz de Confusão (Normalizada)')
plt.savefig(os.path.join(RESULTADOS_DIR, 'confusion_matrix_model1.png'))
plt.close(figure1)

# Gráfico de métricas Modelo 1
plotar_metricas(df_metricas1, "Modelo 1", RESULTADOS_DIR)


# ══════════════════════════════════════════════════════════════
# MODELO 2 — CNN com callback de CM por época (TensorBoard)
# ══════════════════════════════════════════════════════════════

model1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

file_writer = tf.summary.create_file_writer(os.path.join(logdir, 'cm'))


def log_confusion_matrix(epoch, logs):
    test_pred = np.argmax(model1.predict(test_images), axis=1)
    con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=test_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') /
                             con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Modelo 2 — Época {epoch+1}')
    plt.savefig(os.path.join(RESULTADOS_DIR, f'confusion_matrix_model2_epoch_{epoch+1}.png'))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    with file_writer.as_default():
        tf.summary.image("Confusion Matrix", image, step=epoch)


tensorboard_callback2 = tf.keras.callbacks.TensorBoard(log_dir=logdir)
cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

history1 = model1.fit(
    train_images,
    train_labels,
    epochs=5,
    verbose=0,
    callbacks=[tensorboard_callback2, cm_callback],
    validation_data=(test_images, test_labels),
)

# Predições e matriz de confusão — Modelo 2
test_pred2   = np.argmax(model1.predict(test_images), axis=1)
con_mat2_raw = tf.math.confusion_matrix(labels=test_labels, predictions=test_pred2).numpy()
con_mat2_norm = np.around(con_mat2_raw.astype('float') /
                           con_mat2_raw.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat2_df  = pd.DataFrame(con_mat2_norm, index=classes, columns=classes)

# ── Calcula métricas do desafio para Modelo 2 ──
df_metricas2 = calcular_metricas_por_classe(con_mat2_raw, classes)

# Gráfico de métricas Modelo 2
plotar_metricas(df_metricas2, "Modelo 2", RESULTADOS_DIR)


# ══════════════════════════════════════════════════════════════
# GRÁFICO DE COMPARAÇÃO ENTRE MODELOS
# ══════════════════════════════════════════════════════════════

metricas_cols = ["Sensibilidade", "Especificidade", "Acuracia", "Precisao", "F-score"]
media1 = df_metricas1[df_metricas1["Classe"] == "MÉDIA MACRO"][metricas_cols].values[0]
media2 = df_metricas2[df_metricas2["Classe"] == "MÉDIA MACRO"][metricas_cols].values[0]

x = np.arange(len(metricas_cols))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - width/2, media1, width, label="Modelo 1", color="#2196F3", alpha=0.85)
bars2 = ax.bar(x + width/2, media2, width, label="Modelo 2", color="#4CAF50", alpha=0.85)

ax.set_title("Comparação de Métricas — Modelo 1 vs Modelo 2 (Média Macro)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metricas_cols)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Valor")
ax.legend()
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS_DIR, 'comparacao_modelos.png'), dpi=120, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════
# ESCRITA DO ARQUIVO resultados.txt COMPLETO
# ══════════════════════════════════════════════════════════════

TXT_PATH = os.path.join(RESULTADOS_DIR, 'resultados.txt')
with open(TXT_PATH, 'w', encoding='utf-8') as f:

    f.write("=" * 65 + "\n")
    f.write("  MÉTRICAS E RESULTADOS DE AVALIAÇÃO\n")
    f.write("  Cálculo de Métricas de Avaliação de Aprendizado\n")
    f.write("  Formação Machine Learning Specialist\n")
    f.write("=" * 65 + "\n")
    f.write(f"  TensorFlow Version: {tf.__version__}\n")
    f.write("=" * 65 + "\n\n")

    # ── Explicação das métricas ──
    f.write("MÉTRICAS IMPLEMENTADAS (Tabela 1 do Desafio):\n\n")
    f.write("  Sensibilidade  = VP / (VP + FN)\n")
    f.write("    → Mede a capacidade de detectar os positivos reais (Recall)\n\n")
    f.write("  Especificidade = VN / (FP + VN)\n")
    f.write("    → Mede a capacidade de detectar os negativos reais\n\n")
    f.write("  Acurácia       = (VP + VN) / N\n")
    f.write("    → Proporção total de acertos sobre todos os elementos\n\n")
    f.write("  Precisão       = VP / (VP + FP)\n")
    f.write("    → Dos que classificou como positivo, quantos realmente são\n\n")
    f.write("  F-score        = 2 × (Precisão × Sensibilidade) / (Precisão + Sensibilidade)\n")
    f.write("    → Média harmônica entre Precisão e Sensibilidade\n\n")

    # ── MODELO 1 ──
    f.write("=" * 65 + "\n")
    f.write("  MODELO 1: SUMÁRIO DA ARQUITETURA\n")
    f.write("=" * 65 + "\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n")

    f.write("=" * 65 + "\n")
    f.write("  MODELO 1: HISTÓRICO DE TREINAMENTO\n")
    f.write("=" * 65 + "\n")
    for epoch in range(len(history.history['accuracy'])):
        f.write(f"  Epoch {epoch+1}/5:\n")
        f.write(f"    Loss: {history.history['loss'][epoch]:.4f}"
                f" | Accuracy: {history.history['accuracy'][epoch]:.4f}"
                f" | Val Loss: {history.history['val_loss'][epoch]:.4f}"
                f" | Val Accuracy: {history.history['val_accuracy'][epoch]:.4f}\n")
    f.write("\n")

    f.write("=" * 65 + "\n")
    f.write("  MODELO 1: MATRIZ DE CONFUSÃO (NORMALIZADA)\n")
    f.write("=" * 65 + "\n")
    f.write(con_mat1_df.to_string())
    f.write("\n\n")

    # ── Métricas Modelo 1 (núcleo do desafio) ──
    escrever_metricas_txt(f, df_metricas1, "MODELO 1", con_mat1_raw)

    # ── MODELO 2 ──
    f.write("=" * 65 + "\n")
    f.write("  MODELO 2: SUMÁRIO DA ARQUITETURA\n")
    f.write("=" * 65 + "\n")
    model1.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n")

    f.write("=" * 65 + "\n")
    f.write("  MODELO 2: HISTÓRICO DE TREINAMENTO\n")
    f.write("=" * 65 + "\n")
    for epoch in range(len(history1.history['accuracy'])):
        f.write(f"  Epoch {epoch+1}/5:\n")
        f.write(f"    Loss: {history1.history['loss'][epoch]:.4f}"
                f" | Accuracy: {history1.history['accuracy'][epoch]:.4f}"
                f" | Val Loss: {history1.history['val_loss'][epoch]:.4f}"
                f" | Val Accuracy: {history1.history['val_accuracy'][epoch]:.4f}\n")
    f.write("\n")

    f.write("=" * 65 + "\n")
    f.write("  MODELO 2: MATRIZ DE CONFUSÃO FINAL (NORMALIZADA)\n")
    f.write("=" * 65 + "\n")
    f.write(con_mat2_df.to_string())
    f.write("\n\n")

    # ── Métricas Modelo 2 (núcleo do desafio) ──
    escrever_metricas_txt(f, df_metricas2, "MODELO 2", con_mat2_raw)

    # ── Comparação final ──
    f.write("=" * 65 + "\n")
    f.write("  COMPARAÇÃO FINAL — MÉDIA MACRO (Modelo 1 vs Modelo 2)\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"  {'Métrica':<18} {'Modelo 1':>12} {'Modelo 2':>12} {'Diferença':>12}\n")
    f.write("  " + "-" * 56 + "\n")
    for col, v1, v2 in zip(metricas_cols, media1, media2):
        diff = v2 - v1
        sinal = "+" if diff >= 0 else ""
        f.write(f"  {col:<18} {v1:>12.4f} {v2:>12.4f} {sinal+f'{diff:.4f}':>12}\n")
    f.write("\n")
    f.write("  Arquivos gerados na pasta 'resultados/':\n")
    f.write("    • resultados.txt                    (este arquivo)\n")
    f.write("    • confusion_matrix_model1.png       (heatmap Modelo 1)\n")
    f.write("    • confusion_matrix_model2_epoch_*.png (heatmap por época)\n")
    f.write("    • metricas_Modelo_1.png             (gráfico métricas M1)\n")
    f.write("    • metricas_Modelo_2.png             (gráfico métricas M2)\n")
    f.write("    • comparacao_modelos.png            (M1 vs M2)\n")
    f.write("\n" + "=" * 65 + "\n")
    f.write("  Fim do Relatório\n")
    f.write("=" * 65 + "\n")

print(f"\n✔ Concluído! Resultados salvos em: {TXT_PATH}")
print(f"  Gráficos salvos em: {RESULTADOS_DIR}")
print(f"  Logs do TensorBoard em: {logdir}")

# Para iniciar o TensorBoard:
# %tensorboard --logdir log
