"""
=============================================================
  ALGORITMOS DE REGRESSÃO PARA MACHINE LEARNING
=============================================================
Cobre os principais algoritmos:
  1. Regressão Linear (simples e múltipla)
  2. Regressão Polinomial
  3. Ridge (L2) e Lasso (L1)
  4. Elastic Net
  5. Árvore de Decisão
  6. Random Forest
  7. Gradient Boosting (XGBoost-like via sklearn)
  8. SVR (Support Vector Regression)
  9. KNN Regressor
 10. MLP (Rede Neural - MLPRegressor)

Requisitos:
  pip install scikit-learn numpy pandas matplotlib seaborn
"""

import builtins
import sys
import os
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.datasets import make_regression, fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Modelos

# ─────────────────────────────────────────────
# 0. CONFIGURAÇÕES GERAIS E SAÍDA
# ─────────────────────────────────────────────

# Garante que o console tente usar utf-8 para evitar erros de caractere no Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SEED = 42
np.random.seed(SEED)

# Define o diretório de saída relativo ao script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Captura toda a saída impressa para salvar em arquivo
output_lines = []


def custom_print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    file = kwargs.get("file", None)
    if file is None:
        msg = sep.join(map(str, args)) + end
        output_lines.append(msg)
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback robusto para quando o terminal não aceitar certos caracteres
        encoding = sys.stdout.encoding or 'ascii'
        safe_args = [str(arg).encode(encoding, errors='replace').decode(
            encoding) for arg in args]
        builtins.print(*safe_args, **kwargs)


# Sobrescreve print no escopo local
print = custom_print

# ─────────────────────────────────────────────
# 1. CARREGANDO / GERANDO DADOS
# ─────────────────────────────────────────────
print("=" * 60)
print("CARREGANDO DATASET")
print("=" * 60)

# Opção A: Dataset sintético
# X, y = make_regression(n_samples=500, n_features=10, noise=20, random_state=SEED)

# Opção B: Dataset real (California Housing)
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

print(f"Shape dos dados  : {X.shape}")
print(
    f"Variável alvo    : {housing.target_names[0] if hasattr(housing, 'target_names') else 'MedHouseVal'}")
print(f"Média do target  : {y.mean():.3f}")
print(f"Desvio padrão    : {y.std():.3f}\n")

# ─────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Treino : {X_train.shape[0]} amostras")
print(f"Teste  : {X_test.shape[0]} amostras\n")

# ─────────────────────────────────────────────
# 3. FUNÇÃO DE AVALIAÇÃO
# ─────────────────────────────────────────────


def avaliar_modelo(nome, modelo, X_tr, X_te, y_tr, y_te, cv=5):
    """Treina, avalia e retorna métricas do modelo."""
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)

    # Validação cruzada (R²)
    cv_scores = cross_val_score(
        modelo, X_tr, y_tr,
        cv=KFold(n_splits=cv, shuffle=True, random_state=SEED),
        scoring="r2"
    )

    print(f"{'─'*50}")
    print(f"  {nome}")
    print(f"{'─'*50}")
    print(f"  RMSE          : {rmse:.4f}")
    print(f"  MAE           : {mae:.4f}")
    print(f"  R² (teste)    : {r2:.4f}")
    print(
        f"  R² CV ({cv}fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()

    return {
        "Modelo": nome,
        "RMSE":   round(rmse, 4),
        "MAE":    round(mae, 4),
        "R²":     round(r2, 4),
        "R² CV":  round(cv_scores.mean(), 4),
    }, modelo.predict(X_te)


# ─────────────────────────────────────────────
# 4. DEFINIÇÃO DOS MODELOS
# ─────────────────────────────────────────────
print("=" * 60)
print("TREINAMENTO E AVALIAÇÃO DOS MODELOS")
print("=" * 60 + "\n")

modelos_config = [
    # (nome, modelo, usa_scaler)
    ("Regressão Linear",        LinearRegression(
    ),                                 True),
    ("Regressão Polinomial(2)", Pipeline([                                           # Pipeline já escala internamente
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr",     LinearRegression()),
    ]),                                                                              False),
    ("Ridge (α=1.0)",           Ridge(
        alpha=1.0),                                   True),
    ("Lasso (α=0.01)",          Lasso(
        alpha=0.01, max_iter=5000),                   True),
    ("Elastic Net",             ElasticNet(
        alpha=0.01, l1_ratio=0.5, max_iter=5000), True),
    ("Árvore de Decisão",       DecisionTreeRegressor(
        max_depth=6, random_state=SEED), False),
    ("Random Forest",           RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=SEED, n_jobs=-1), False),
    ("Gradient Boosting",       GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=SEED), False),
    ("SVR (RBF)",               SVR(kernel="rbf",
     C=10, epsilon=0.1),               True),
    ("KNN (k=5)",               KNeighborsRegressor(
        n_neighbors=5, n_jobs=-1),      True),
    ("MLP (Rede Neural)",       MLPRegressor(hidden_layer_sizes=(
        128, 64), max_iter=500, random_state=SEED), True),
]

resultados = []
predicoes = {}

for nome, modelo, usar_sc in modelos_config:
    Xtr = X_train_sc if usar_sc else X_train
    Xte = X_test_sc if usar_sc else X_test

    res, y_pred = avaliar_modelo(nome, modelo, Xtr, Xte, y_train, y_test)
    resultados.append(res)
    predicoes[nome] = y_pred

# ─────────────────────────────────────────────
# 5. TABELA COMPARATIVA
# ─────────────────────────────────────────────
df_res = pd.DataFrame(resultados).sort_values(
    "R²", ascending=False).reset_index(drop=True)
df_res.index += 1

print("\n" + "=" * 60)
print("RANKING DOS MODELOS (por R²)")
print("=" * 60)
print(df_res.to_string())

# ─────────────────────────────────────────────
# 6. VISUALIZAÇÕES
# ─────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")

# 6.1 Barras comparativas
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Comparação dos Algoritmos de Regressão",
             fontsize=16, fontweight="bold")

metrics = ["R²", "RMSE", "MAE"]
for ax, metric in zip(axes, metrics):
    dados = df_res.sort_values(metric, ascending=(metric != "R²"))
    cores = sns.color_palette("viridis", len(dados))
    bars = ax.barh(dados["Modelo"], dados[metric], color=cores)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_xlabel(metric)
    for bar, val in zip(bars, dados[metric]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
comparacao_path = os.path.join(output_dir, "comparacao_modelos.png")
plt.savefig(comparacao_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nGráfico salvo: {comparacao_path}")

# 6.2 Real vs Previsto (top 3 modelos)
top3 = df_res.head(3)["Modelo"].tolist()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Real vs Previsto — Top 3 Modelos",
             fontsize=15, fontweight="bold")

for ax, nome in zip(axes, top3):
    y_pred = predicoes[nome]
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=2, label="Perfeito")
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Previsto")
    r2 = df_res[df_res["Modelo"] == nome]["R²"].values[0]
    ax.set_title(f"{nome}\nR² = {r2:.4f}", fontsize=11)
    ax.legend(fontsize=8)

plt.tight_layout()
real_vs_previsto_path = os.path.join(output_dir, "real_vs_previsto.png")
plt.savefig(real_vs_previsto_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Gráfico salvo: {real_vs_previsto_path}")

# 6.3 Distribuição dos resíduos (melhor modelo)
melhor = df_res.iloc[0]["Modelo"]
residuos = y_test.values - predicoes[melhor]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Análise de Resíduos — {melhor}", fontsize=14, fontweight="bold")

axes[0].hist(residuos, bins=50, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_title("Distribuição dos Resíduos")
axes[0].set_xlabel("Resíduo")

axes[1].scatter(predicoes[melhor], residuos,
                alpha=0.3, s=10, color="steelblue")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Resíduos vs Previstos")
axes[1].set_xlabel("Valor Previsto")
axes[1].set_ylabel("Resíduo")

plt.tight_layout()
residuos_path = os.path.join(output_dir, "residuos.png")
plt.savefig(residuos_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Gráfico salvo: {residuos_path}")

# ─────────────────────────────────────────────
# 7. IMPORTÂNCIA DAS FEATURES (Random Forest)
# ─────────────────────────────────────────────
rf_model = next(m for n, m, _ in modelos_config if "Random Forest" in n)

importancias = pd.Series(
    rf_model.feature_importances_,
    index=X.columns if hasattr(X, "columns") else [
        f"f{i}" for i in range(X.shape[1])]
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
importancias.plot.barh(ax=ax, color=sns.color_palette(
    "viridis", len(importancias)))
ax.set_title("Importância das Features — Random Forest",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importância Relativa")
plt.tight_layout()
feature_importance_path = os.path.join(output_dir, "feature_importance.png")
plt.savefig(feature_importance_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Gráfico salvo: {feature_importance_path}\n")

# ─────────────────────────────────────────────
# 8. CONCLUSÃO
# ─────────────────────────────────────────────
print("=" * 60)
print("CONCLUSÃO")
print("=" * 60)
print(f"\n  Melhor modelo  : {melhor}")
print(f"  R²             : {df_res.iloc[0]['R²']}")
print(f"  RMSE           : {df_res.iloc[0]['RMSE']}")
print(f"  MAE            : {df_res.iloc[0]['MAE']}")
print("\n  Arquivos gerados:")
print(f"    - {comparacao_path}")
print(f"    - {real_vs_previsto_path}")
print(f"    - {residuos_path}")
print(f"    - {feature_importance_path}")
print("\nScript finalizado com sucesso!")

# Salvar a saída capturada em resultados_regressao.txt
resultados_path = os.path.join(output_dir, "resultados_regressao.txt")
with open(resultados_path, "w", encoding="utf-8") as f:
    f.writelines(output_lines)

builtins.print(f"\n[OK] Resultados em texto salvos em: {resultados_path}")
