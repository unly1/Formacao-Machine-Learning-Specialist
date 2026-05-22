"""
Script de Treinamento Supervisionado - Classificação
Framework: Scikit-learn
"""

import io
import sys
import contextlib
from datetime import datetime
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# Pasta de saída: subpasta "resultado" criada automaticamente
OUTPUT_DIR = Path(__file__).parent / "resultado"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Buffer para capturar todo o output e salvar no .txt
_buffer = io.StringIO()


class _Tee:
    """Redireciona o print para o terminal E para o buffer ao mesmo tempo."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = _Tee(sys.__stdout__, _buffer)

# ─────────────────────────────────────────
# 1. CARREGAR DADOS
# ─────────────────────────────────────────
# Substitua esta parte pelos seus próprios dados, por exemplo:
# import pandas as pd
# df = pd.read_csv("seus_dados.csv")
# X = df.drop(columns=["target"]).values
# y = df["target"].values

data = load_iris()
X, y = data.data, data.target
class_names = data.target_names  # ex: ['setosa', 'versicolor', 'virginica']

print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"Classes: {class_names}\n")

# ─────────────────────────────────────────
# 2. DIVISÃO TREINO / TESTE
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% para teste
    random_state=42,
    stratify=y,         # mantém proporção de classes
)

print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras\n")

# ─────────────────────────────────────────
# 3. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit apenas no treino
X_test = scaler.transform(X_test)       # aplica a mesma escala no teste

# ─────────────────────────────────────────
# 4. DEFINIR E TREINAR O MODELO
# ─────────────────────────────────────────
# Troque por outro classificador se quiser, exemplos:
# from sklearn.linear_model import LogisticRegression  → LogisticRegression()
# from sklearn.svm import SVC                          → SVC(kernel='rbf')
# from sklearn.neighbors import KNeighborsClassifier   → KNeighborsClassifier(n_neighbors=5)

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
)

print("Treinando o modelo...")
modelo.fit(X_train, y_train)
print("Treinamento concluído!\n")

# ─────────────────────────────────────────
# 5. AVALIAÇÃO
# ─────────────────────────────────────────
y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {acuracia * 100:.2f}%\n")

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────
# 6. SALVAR O MODELO
# ─────────────────────────────────────────
path_modelo = OUTPUT_DIR / "modelo_classificacao.pkl"
path_scaler = OUTPUT_DIR / "scaler.pkl"

joblib.dump(modelo, path_modelo)
joblib.dump(scaler, path_scaler)
print(f"\nModelo salvo em '{path_modelo}'")
print(f"Scaler salvo em '{path_scaler}'")

# ─────────────────────────────────────────
# 7. EXEMPLO DE INFERÊNCIA
# ─────────────────────────────────────────
# modelo_carregado = joblib.load(SCRIPT_DIR / "modelo_classificacao.pkl")
# scaler_carregado = joblib.load(SCRIPT_DIR / "scaler.pkl")
# nova_amostra = np.array([[5.1, 3.5, 1.4, 0.2]])
# nova_amostra_scaled = scaler_carregado.transform(nova_amostra)
# predicao = modelo_carregado.predict(nova_amostra_scaled)
# print(f"Classe predita: {class_names[predicao[0]]}")

# ─────────────────────────────────────────
# 8. SALVAR RESULTADO EM .TXT
# ─────────────────────────────────────────
sys.stdout = sys.__stdout__  # restaura o stdout original

path_resultado = OUTPUT_DIR / "resultado_classificacao.txt"
conteudo = _buffer.getvalue()

with open(path_resultado, "w", encoding="utf-8") as f:
    f.write(
        f"Resultado gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(conteudo)

print(f"Resultado salvo em '{path_resultado}'")
