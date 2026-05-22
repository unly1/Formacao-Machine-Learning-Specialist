"""
=============================================================
  EXTRAÇÃO DE FEATURES E REDUÇÃO DE DIMENSIONALIDADE
=============================================================
Técnicas cobertas:
  EXTRAÇÃO / ENGENHARIA DE FEATURES
    1. PolynomialFeatures
    2. Interações manuais
    3. Estatísticas de janela (rolling)

  REDUÇÃO DE DIMENSIONALIDADE (Linear)
    4. PCA  — Principal Component Analysis
    5. LDA  — Linear Discriminant Analysis (supervisionado)
    6. SVD  — Truncated SVD (TF-IDF / texto)
    7. ICA  — Independent Component Analysis

  REDUÇÃO DE DIMENSIONALIDADE (Não-linear / Manifold)
    8. t-SNE
    9. UMAP
   10. Isomap
   11. LLE — Locally Linear Embedding
   12. Autoencoder (Keras/TensorFlow) — opcional

  SELEÇÃO DE FEATURES
   13. Variância baixa (VarianceThreshold)
   14. Correlação alta (manual)
   15. SelectKBest (F-score / MI)
   16. Importância por Random Forest (RFE)
   17. RFECV — Recursive Feature Elimination c/ CV

Requisitos:
  pip install scikit-learn numpy pandas matplotlib seaborn umap-learn
  (opcional) pip install tensorflow  ← para o Autoencoder
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    RFECV,
)
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_digits, load_wine, make_classification
import sys
import os
import atexit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ── Extração

# ── Redução linear

# ── Redução não-linear

# ── Seleção

# Logger customizado para redirecionar o stdout para console e arquivo .txt

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Fallback seguro para o terminal se não suportar algum caractere Unicode (ex: emojis/símbolos)
            encoding = getattr(self.terminal, 'encoding', 'ascii') or 'ascii'
            self.terminal.write(message.encode(
                encoding, errors='replace').decode(encoding))

        if hasattr(self, 'log') and not self.log.closed:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if hasattr(self, 'log') and not self.log.closed:
            self.log.flush()


# Define caminhos e configura redirecionamento de logs
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "resultados_features_dimensionalidade.txt")
sys.stdout = DualLogger(log_file)

# Garante o fechamento limpo do arquivo de log ao encerrar
atexit.register(lambda: sys.stdout.log.close() if hasattr(
    sys.stdout, 'log') and not sys.stdout.log.closed else None)

# Cria pasta de outputs se não existir
outputs_dir = os.path.join(script_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Tentar importar UMAP (opcional)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠  umap-learn não instalado. Pulando UMAP.")

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="darkgrid", palette="tab10")

# ═══════════════════════════════════════════════════════════
# 0. DATASETS
# ═══════════════════════════════════════════════════════════
print("=" * 62)
print("  CARREGANDO DATASETS")
print("=" * 62)

# Digits: 1797 amostras, 64 features (imagens 8x8), 10 classes
digits = load_digits()
X_dig, y_dig = digits.data, digits.target

# Wine: 178 amostras, 13 features, 3 classes
wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# Sintético grande: 2000 amostras, 30 features, 2 classes
X_syn, y_syn = make_classification(
    n_samples=2000, n_features=30, n_informative=10,
    n_redundant=8, n_clusters_per_class=2, random_state=SEED
)

scaler = StandardScaler()
X_dig_sc = scaler.fit_transform(X_dig)
X_wine_sc = scaler.fit_transform(X_wine)
X_syn_sc = scaler.fit_transform(X_syn)

print(f"Digits : {X_dig.shape}  | classes: {np.unique(y_dig)}")
print(f"Wine   : {X_wine.shape} | classes: {np.unique(y_wine)}")
print(f"Sintético: {X_syn.shape} | classes: {np.unique(y_syn)}\n")

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════


def plot_2d(ax, X2, y, titulo, cmap="tab10"):
    """Scatter 2D colorido por classe."""
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap=cmap,
                    alpha=0.6, s=12, linewidths=0)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    return sc


def sep_bar(nome, X_orig, X_red, y, clf=None):
    """Treina LR simples e imprime acurácia pré/pós redução."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    sc_orig = cross_val_score(lr, X_orig, y, cv=5, scoring="accuracy").mean()
    sc_red = cross_val_score(lr, X_red,  y, cv=5, scoring="accuracy").mean()
    print(f"  {nome:30s} | original: {sc_orig:.3f}  reduzido: {sc_red:.3f}")


# ═══════════════════════════════════════════════════════════
# 1. ENGENHARIA DE FEATURES
# ═══════════════════════════════════════════════════════════
print("=" * 62)
print("  1. ENGENHARIA / EXTRAÇÃO DE FEATURES")
print("=" * 62)

# 1.1 PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_wine_sc)
print(f"\n[PolynomialFeatures grau=2]")
print(f"  Wine original : {X_wine_sc.shape[1]:>3} features")
print(f"  Após Poly     : {X_poly.shape[1]:>3} features")

# 1.2 Interações manuais (produto entre pares)
df_wine = pd.DataFrame(X_wine_sc, columns=wine.feature_names)
for i in range(len(wine.feature_names)):
    for j in range(i+1, len(wine.feature_names)):
        fn = wine.feature_names
        df_wine[f"{fn[i]}_x_{fn[j]}"] = df_wine[fn[i]] * df_wine[fn[j]]

print(f"\n[Interações manuais par-a-par]")
print(f"  Features após interação: {df_wine.shape[1]}")

# 1.3 Estatísticas em janela (exemplo com série temporal sintética)
ts = pd.Series(np.cumsum(np.random.randn(200)))
df_ts = pd.DataFrame({"valor": ts})
df_ts["rolling_mean_5"] = ts.rolling(5).mean()
df_ts["rolling_std_5"] = ts.rolling(5).std()
df_ts["rolling_min_10"] = ts.rolling(10).min()
df_ts["rolling_max_10"] = ts.rolling(10).max()
df_ts["lag_1"] = ts.shift(1)
df_ts["lag_2"] = ts.shift(2)
df_ts.dropna(inplace=True)
print(f"\n[Rolling features — série temporal]")
print(df_ts.head(3).to_string())

# ═══════════════════════════════════════════════════════════
# 2. REDUÇÃO LINEAR
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  2. REDUÇÃO DE DIMENSIONALIDADE — LINEAR")
print("=" * 62)

# ── 2.1 PCA
pca_full = PCA(random_state=SEED).fit(X_dig_sc)
var_exp = np.cumsum(pca_full.explained_variance_ratio_)
n95 = np.searchsorted(var_exp, 0.95) + 1

pca2 = PCA(n_components=2, random_state=SEED)
X_pca = pca2.fit_transform(X_dig_sc)

pca_n95 = PCA(n_components=n95, random_state=SEED)
X_pca_n95 = pca_n95.fit_transform(X_dig_sc)

print(f"\n[PCA — Digits 64 features]")
print(f"  Componentes para 95% variância: {n95}")
print(
    f"  Variância explicada (2 CP)    : {pca2.explained_variance_ratio_.sum():.2%}")
sep_bar("PCA (n95)", X_dig_sc, X_pca_n95, y_dig)

# ── 2.2 LDA
lda2 = LDA(n_components=2)
X_lda = lda2.fit_transform(X_wine_sc, y_wine)
print(f"\n[LDA — Wine 13→2]")
print(f"  Variância explicada : {lda2.explained_variance_ratio_.sum():.2%}")
sep_bar("LDA (2 comp)", X_wine_sc, X_lda, y_wine)

# ── 2.3 TruncatedSVD
svd2 = TruncatedSVD(n_components=2, random_state=SEED)
X_svd = svd2.fit_transform(X_dig_sc)
print(f"\n[TruncatedSVD — Digits 64→2]")
print(f"  Variância explicada : {svd2.explained_variance_ratio_.sum():.2%}")

# ── 2.4 ICA
ica2 = FastICA(n_components=2, random_state=SEED, max_iter=500)
X_ica = ica2.fit_transform(X_dig_sc)
print(f"\n[ICA — Digits 64→2]  (componentes independentes)")

# ═══════════════════════════════════════════════════════════
# 3. REDUÇÃO NÃO-LINEAR (MANIFOLD)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  3. REDUÇÃO DE DIMENSIONALIDADE — NÃO-LINEAR")
print("=" * 62)

# Sub-amostra para agilidade
idx = np.random.choice(len(X_dig_sc), 600, replace=False)
X_sub = X_dig_sc[idx]
y_sub = y_dig[idx]

# Reduz para 10D via PCA antes dos métodos de manifold (boa prática)
X_pre = PCA(n_components=20, random_state=SEED).fit_transform(X_sub)

print("\nAplicando técnicas de manifold (sub-amostra 600)...")

tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=500)
X_tsne = tsne.fit_transform(X_pre)
print("  t-SNE   ✓")

iso = Isomap(n_components=2, n_neighbors=10)
X_iso = iso.fit_transform(X_pre)
print("  Isomap  ✓")

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=SEED)
X_lle = lle.fit_transform(X_pre)
print("  LLE     ✓")

X_umap = None
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, random_state=SEED)
    X_umap = reducer.fit_transform(X_pre)
    print("  UMAP    ✓")

# ═══════════════════════════════════════════════════════════
# 4. SELEÇÃO DE FEATURES
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  4. SELEÇÃO DE FEATURES")
print("=" * 62)

# ── 4.1 Variância baixa
vt = VarianceThreshold(threshold=0.1)
X_vt = vt.fit_transform(X_syn_sc)
print(f"\n[VarianceThreshold=0.1]")
print(f"  {X_syn_sc.shape[1]} → {X_vt.shape[1]} features")

# ── 4.2 Correlação alta
df_corr = pd.DataFrame(X_syn, columns=[f"f{i}" for i in range(X_syn.shape[1])])
corr_mat = df_corr.corr().abs()
upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
drop_corr = [c for c in upper.columns if any(upper[c] > 0.85)]
df_low_corr = df_corr.drop(columns=drop_corr)
print(f"\n[Correlação > 0.85 removida]")
print(
    f"  {X_syn.shape[1]} → {df_low_corr.shape[1]} features  (removidas: {len(drop_corr)})")

# ── 4.3 SelectKBest — F-score
skb_f = SelectKBest(score_func=f_classif, k=10)
X_kbf = skb_f.fit_transform(X_syn_sc, y_syn)
print(f"\n[SelectKBest F-score k=10]")
print(f"  {X_syn_sc.shape[1]} → {X_kbf.shape[1]} features")
print(f"  Top features: {np.where(skb_f.get_support())[0].tolist()}")

# ── 4.4 SelectKBest — Mutual Information
skb_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_kbmi = skb_mi.fit_transform(X_syn_sc, y_syn)
print(f"\n[SelectKBest Mutual Information k=10]")
print(f"  Top features: {np.where(skb_mi.get_support())[0].tolist()}")

# ── 4.5 RFE com Random Forest
rf_rfe = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
rfe = RFE(estimator=rf_rfe, n_features_to_select=10, step=2)
rfe.fit(X_syn_sc, y_syn)
print(f"\n[RFE — Random Forest, 10 features]")
print(f"  Selecionadas: {np.where(rfe.support_)[0].tolist()}")

# ── 4.6 RFECV
lr_rfe = LogisticRegression(max_iter=1000, random_state=SEED)
rfecv = RFECV(estimator=lr_rfe,
              cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
              scoring="accuracy", step=1, min_features_to_select=5, n_jobs=-1)
rfecv.fit(X_syn_sc, y_syn)
print(f"\n[RFECV — Logistic Regression]")
print(f"  Nº ótimo de features: {rfecv.n_features_}")
print(f"  Features selecionadas: {np.where(rfecv.support_)[0].tolist()}")

# ═══════════════════════════════════════════════════════════
# 5. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  5. GERANDO VISUALIZAÇÕES")
print("=" * 62)

# ── Fig 1: Variância Explicada PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PCA — Variância Explicada (Digits)",
             fontsize=14, fontweight="bold")

axes[0].plot(range(1, len(var_exp)+1), var_exp,
             marker="o", markersize=3, color="steelblue")
axes[0].axhline(0.95, color="red", linestyle="--", label="95%")
axes[0].axvline(n95, color="orange", linestyle="--", label=f"n={n95}")
axes[0].set_xlabel("Nº de Componentes")
axes[0].set_ylabel("Variância Acumulada")
axes[0].set_title("Variância Acumulada")
axes[0].legend()

axes[1].bar(range(1, 21), pca_full.explained_variance_ratio_[
            :20], color="steelblue")
axes[1].set_xlabel("Componente")
axes[1].set_ylabel("Variância Explicada")
axes[1].set_title("Variância por Componente (top 20)")

plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "pca_variancia.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  pca_variancia.png ✓")

# ── Fig 2: Comparativo métodos lineares 2D
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("Redução Linear — 2D (Digits / Wine)",
             fontsize=14, fontweight="bold")

plot_2d(axes[0], X_pca,             y_dig, "PCA (Digits)")
plot_2d(axes[1], X_lda,             y_wine,             "LDA (Wine)")
plot_2d(axes[2], X_svd,             y_dig, "TruncSVD (Digits)")
plot_2d(axes[3], X_ica,             y_dig, "ICA (Digits)")

plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "reducao_linear.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  reducao_linear.png ✓")

# ── Fig 3: Métodos de manifold
n_methods = 3 + (1 if HAS_UMAP else 0)
fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
if n_methods == 1:
    axes = [axes]
fig.suptitle("Redução Não-Linear — Manifold (Digits, 600 amostras)",
             fontsize=14, fontweight="bold")

manifolds = [("t-SNE", X_tsne), ("Isomap", X_iso), ("LLE", X_lle)]
if HAS_UMAP:
    manifolds.append(("UMAP", X_umap))

for ax, (nome, X2) in zip(axes, manifolds):
    plot_2d(ax, X2, y_sub, nome)

plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "manifold.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  manifold.png ✓")

# ── Fig 4: Scores de seleção de features
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Scores de Seleção de Features (Sintético)",
             fontsize=14, fontweight="bold")

feat_names = [f"f{i}" for i in range(X_syn_sc.shape[1])]

# F-score
f_scores = pd.Series(
    skb_f.scores_, index=feat_names).sort_values(ascending=True)
colors_f = ["tomato" if i in np.where(skb_f.get_support())[0] else "steelblue"
            for i in range(len(feat_names))]
f_scores_sorted_idx = f_scores.index
colors_sorted = [colors_f[int(n[1:])] for n in f_scores_sorted_idx]
axes[0].barh(f_scores_sorted_idx, f_scores.values, color=colors_sorted)
axes[0].set_title("F-score (vermelho = selecionada)")
axes[0].set_xlabel("Score")

# Mutual Information
mi_scores = pd.Series(
    skb_mi.scores_, index=feat_names).sort_values(ascending=True)
colors_mi = ["tomato" if i in np.where(skb_mi.get_support())[0] else "steelblue"
             for i in range(len(feat_names))]
mi_sorted_idx = mi_scores.index
colors_mi_sorted = [colors_mi[int(n[1:])] for n in mi_sorted_idx]
axes[1].barh(mi_sorted_idx, mi_scores.values, color=colors_mi_sorted)
axes[1].set_title("Mutual Information (vermelho = selecionada)")
axes[1].set_xlabel("Score")

plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "feature_selection_scores.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  feature_selection_scores.png ✓")

# ── Fig 5: RFECV — Nº de features vs Acurácia
fig, ax = plt.subplots(figsize=(10, 5))
n_feat_range = range(rfecv.min_features_to_select,
                     rfecv.min_features_to_select + len(rfecv.cv_results_["mean_test_score"]))
ax.plot(n_feat_range, rfecv.cv_results_["mean_test_score"],
        marker="o", markersize=4, color="steelblue")
ax.fill_between(n_feat_range,
                rfecv.cv_results_["mean_test_score"] -
                rfecv.cv_results_["std_test_score"],
                rfecv.cv_results_["mean_test_score"] +
                rfecv.cv_results_["std_test_score"],
                alpha=0.2, color="steelblue")
ax.axvline(rfecv.n_features_, color="red", linestyle="--",
           label=f"Ótimo: {rfecv.n_features_} features")
ax.set_xlabel("Nº de Features")
ax.set_ylabel("Acurácia (CV)")
ax.set_title("RFECV — Nº de Features vs Acurácia",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "rfecv.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  rfecv.png ✓")

# ── Fig 6: Heatmap de correlação (Sintético, top 15 features)
fig, ax = plt.subplots(figsize=(12, 10))
corr_top = pd.DataFrame(X_syn[:, :15], columns=[
                        f"f{i}" for i in range(15)]).corr()
mask = np.triu(np.ones_like(corr_top, dtype=bool))
sns.heatmap(corr_top, mask=mask, cmap="coolwarm", center=0,
            annot=True, fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Matriz de Correlação (15 features sintéticas)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "correlacao.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  correlacao.png ✓")

# ═══════════════════════════════════════════════════════════
# 6. RESUMO FINAL
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  RESUMO")
print("=" * 62)

resumo = pd.DataFrame([
    {"Técnica": "PolynomialFeatures (grau 2)", "Entrada": 13,
     "Saída": X_poly.shape[1], "Tipo": "Extração"},
    {"Técnica": "PCA (95% var)", "Entrada": 64,
     "Saída": n95, "Tipo": "Red. Linear"},
    {"Técnica": "LDA", "Entrada": 13, "Saída": 2, "Tipo": "Red. Linear (sup)"},
    {"Técnica": "TruncSVD", "Entrada": 64, "Saída": 2, "Tipo": "Red. Linear"},
    {"Técnica": "ICA", "Entrada": 64, "Saída": 2, "Tipo": "Red. Linear"},
    {"Técnica": "t-SNE", "Entrada": 20, "Saída": 2, "Tipo": "Red. Não-linear"},
    {"Técnica": "Isomap", "Entrada": 20, "Saída": 2, "Tipo": "Red. Não-linear"},
    {"Técnica": "LLE", "Entrada": 20, "Saída": 2, "Tipo": "Red. Não-linear"},
    {"Técnica": "VarianceThreshold", "Entrada": 30,
        "Saída": X_vt.shape[1], "Tipo": "Seleção"},
    {"Técnica": "Correlação Alta", "Entrada": 30,
        "Saída": df_low_corr.shape[1], "Tipo": "Seleção"},
    {"Técnica": "SelectKBest F-score", "Entrada": 30, "Saída": 10, "Tipo": "Seleção"},
    {"Técnica": "SelectKBest MI", "Entrada": 30, "Saída": 10, "Tipo": "Seleção"},
    {"Técnica": "RFE (Random Forest)", "Entrada": 30,
     "Saída": 10, "Tipo": "Seleção"},
    {"Técnica": "RFECV (LogReg)", "Entrada": 30,
     "Saída": rfecv.n_features_, "Tipo": "Seleção"},
])
if HAS_UMAP:
    resumo = pd.concat([resumo, pd.DataFrame([{
        "Técnica": "UMAP", "Entrada": 20, "Saída": 2, "Tipo": "Red. Não-linear"
    }])], ignore_index=True)

print(resumo.to_string(index=False))
print(f"\nResultados salvos em: {log_file}")
print(f"Figuras salvas no diretório: {outputs_dir}")
print("\nArquivos de imagem gerados:")
for f in ["pca_variancia.png", "reducao_linear.png", "manifold.png",
          "feature_selection_scores.png", "rfecv.png", "correlacao.png"]:
    print(f"  • {f}")
print("\nScript finalizado com sucesso!")
