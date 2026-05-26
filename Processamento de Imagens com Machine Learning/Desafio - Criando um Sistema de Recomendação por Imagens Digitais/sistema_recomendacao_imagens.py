# -*- coding: utf-8 -*-
"""
Sistema de Recomendação por Similaridade de Imagens
====================================================
Uso:
    # 1. Construir o índice (primeira vez ou para reconstruir):
    python sistema_recomendacao_imagens.py --mode build

    # 2. Recomendar produtos similares a uma imagem:
    python sistema_recomendacao_imagens.py --mode recommend --image caminho/para/imagem.jpg

    # 3. Avaliar precisão por categoria:
    python sistema_recomendacao_imagens.py --mode evaluate

    # 4. Adicionar um novo produto ao índice:
    python sistema_recomendacao_imagens.py --mode add --image nova_img.jpg --id 999999

    # 5. Remover um produto do índice:
    python sistema_recomendacao_imagens.py --mode remove --id 12345
"""

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import os
import sys

# Garante saída UTF-8 no terminal Windows (evita UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import json
import pickle
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sem janela (salva em arquivo)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES — ajuste conforme necessário
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(r"C:\Users\moliv\Documents\Formacao-Machine-Learning-Specialist\Processamento de Imagens com Machine Learning\Desafio - Criando um Sistema de Recomendação por Imagens Digitais")

DATA_DIR = BASE_DIR / "archive"       # pasta com images/ e styles.csv (baixada localmente)
IMAGES_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "styles.csv"

CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "modelo_recomendacao"
RESULTS_DIR = BASE_DIR / "resultados"

IMG_SIZE = (224, 224)
MAX_IMAGES = 50        # None = todas; reduza para testes rápidos (ex: 500)
BATCH_SIZE = 32
MODEL_BACKBONE = "resnet50"   # "resnet50" ou "vgg16"
TOP_N = 5             # número de recomendações

# Caminhos dos arquivos salvos
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
PRODUCTS_FILE = OUTPUT_DIR / "produtos.csv"
PCA_FILE = OUTPUT_DIR / "pca.pkl"
CONFIG_FILE = OUTPUT_DIR / "config.json"

# ─────────────────────────────────────────────────────────────────────────────
# SETUP DE PASTAS
# ─────────────────────────────────────────────────────────────────────────────

for d in [CACHE_DIR, OUTPUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICAÇÃO DO DATASET LOCAL
# ─────────────────────────────────────────────────────────────────────────────

def verificar_dataset():
    """Verifica se a base de dados já foi colocada na pasta archive/"""
    if not (IMAGES_DIR.exists() and CSV_PATH.exists()):
        print(f"\n[ERRO] Base de dados não encontrada em: {DATA_DIR}")
        print("Certifique-se de que a pasta 'archive/' contém:")
        print("  - A subpasta 'images/' com as imagens do produto (.jpg)")
        print("  - O arquivo 'styles.csv' com os metadados dos produtos")
        print("\nComo você já baixou os dados por fora, basta extrair a pasta")
        print(f"'images' e o arquivo 'styles.csv' diretamente em: {DATA_DIR}\n")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DO MODELO
# ─────────────────────────────────────────────────────────────────────────────

def load_model(backbone: str):
    """
    Carrega o modelo de extração de features (ResNet50 ou VGG16).
    Importação feita aqui para não travar o script se TF não estiver instalado
    ao usar apenas as funções de gerenciamento do índice.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50, VGG16
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pp
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pp

    print(f"TensorFlow {tf.__version__} — carregando {backbone.upper()}...")

    if backbone == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        preprocess_fn = resnet_pp
    elif backbone == "vgg16":
        base = VGG16(weights="imagenet", include_top=False, pooling="avg")
        preprocess_fn = vgg_pp
    else:
        raise ValueError(
            f"Backbone não suportado: {backbone}. Use 'resnet50' ou 'vgg16'.")

    base.trainable = False
    print(f"  Dimensão do embedding: {base.output_shape[-1]}")
    return base, preprocess_fn


# ─────────────────────────────────────────────────────────────────────────────
# PRÉ-PROCESSAMENTO DE IMAGEM
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path, preprocess_fn):
    from tensorflow.keras.preprocessing import image as kimage
    img = kimage.load_img(str(image_path), target_size=IMG_SIZE)
    arr = kimage.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_fn(arr)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DO DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"[ERRO] styles.csv não encontrado em: {CSV_PATH}")
        print("Baixe o dataset em: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    df["image_path"] = df["id"].apply(lambda x: IMAGES_DIR / f"{x}.jpg")
    df["exists"] = df["image_path"].apply(lambda p: p.exists())
    df = df[df["exists"]].reset_index(drop=True)

    if MAX_IMAGES:
        df = df.head(MAX_IMAGES)

    print(f"Dataset: {len(df)} produtos carregados")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXTRAÇÃO DE FEATURES (CONSTRUÇÃO DO ÍNDICE)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(df, model, preprocess_fn):
    embeddings = []
    valid_indices = []
    paths = df["image_path"].tolist()

    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Extraindo features"):
        batch_paths = paths[i: i + BATCH_SIZE]
        batch_imgs = []
        batch_idxs = []

        for j, path in enumerate(batch_paths):
            try:
                arr = preprocess_image(path, preprocess_fn)
                batch_imgs.append(arr[0])
                batch_idxs.append(i + j)
            except Exception:
                pass  # imagem corrompida — ignora

        if batch_imgs:
            batch_arr = np.array(batch_imgs)
            feats = model.predict(batch_arr, verbose=0)
            embeddings.extend(feats)
            valid_indices.extend(batch_idxs)

    embeddings_arr = normalize(np.array(embeddings))
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    return embeddings_arr, valid_df


def build_index():
    """Constrói o banco de embeddings do zero e salva em OUTPUT_DIR."""
    print("\n=== CONSTRUINDO ÍNDICE ===")
    df = load_dataset()

    # Checa cache para não reprocessar
    cache_file = CACHE_DIR / f"raw_{MODEL_BACKBONE}_{len(df)}.pkl"
    if cache_file.exists():
        print(f"Cache encontrado: {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        embeddings = cached["embeddings"]
        valid_df = cached["df"]
    else:
        model, preprocess_fn = load_model(MODEL_BACKBONE)
        embeddings, valid_df = extract_all_features(df, model, preprocess_fn)
        with open(cache_file, "wb") as f:
            pickle.dump({"embeddings": embeddings, "df": valid_df}, f)
        print(f"Cache salvo: {cache_file}")

    print(f"Embeddings: {embeddings.shape}")

    # PCA opcional (comprime 2048 → 512 mantendo ~95% da variância)
    N_COMPONENTS = min(512, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    embeddings_pca = normalize(pca.fit_transform(embeddings))
    explained = pca.explained_variance_ratio_.sum()
    print(
        f"PCA: {embeddings.shape[1]}D → {N_COMPONENTS}D  ({explained:.1%} variância mantida)")

    # Salva artefatos
    np.save(EMBEDDINGS_FILE, embeddings_pca)
    valid_df.to_csv(PRODUCTS_FILE, index=False)
    with open(PCA_FILE, "wb") as f:
        pickle.dump(pca, f)

    config = {
        "backbone":        MODEL_BACKBONE,
        "img_size":        list(IMG_SIZE),
        "n_components_pca": N_COMPONENTS,
        "n_products":      len(valid_df),
        "max_images":      MAX_IMAGES,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nÍndice salvo em: {OUTPUT_DIR}")
    print(f"Total de produtos indexados: {len(valid_df)}")
    return embeddings_pca, valid_df


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DO ÍNDICE EXISTENTE
# ─────────────────────────────────────────────────────────────────────────────

def load_index():
    """Carrega embeddings e metadados salvos do disco."""
    if not EMBEDDINGS_FILE.exists() or not PRODUCTS_FILE.exists():
        print("[ERRO] Índice não encontrado. Rode primeiro com --mode build")
        sys.exit(1)

    embeddings = np.load(EMBEDDINGS_FILE)
    valid_df = pd.read_csv(PRODUCTS_FILE)
    valid_df["image_path"] = valid_df["id"].apply(
        lambda x: IMAGES_DIR / f"{x}.jpg")
    print(
        f"Índice carregado: {embeddings.shape[0]} produtos, {embeddings.shape[1]}D")
    return embeddings, valid_df


def load_pca():
    if not PCA_FILE.exists():
        return None
    with open(PCA_FILE, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# RECOMENDAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendations(query_image_path, embeddings, valid_df, model, preprocess_fn, pca=None, top_n=TOP_N):
    """
    Retorna os top_n produtos mais similares visualmente à imagem de consulta.
    """
    arr = preprocess_image(query_image_path, preprocess_fn)
    query_feat = model.predict(arr, verbose=0)
    query_feat = normalize(query_feat)

    if pca is not None:
        query_feat = normalize(pca.transform(query_feat))

    scores = cosine_similarity(query_feat, embeddings)[0]

    # Exclui a própria imagem se estiver no índice
    query_id = str(Path(query_image_path).stem)
    sorted_idx = np.argsort(scores)[::-1]
    filtered = [i for i in sorted_idx if str(
        valid_df.iloc[i]["id"]) != query_id][:top_n]

    recs = valid_df.iloc[filtered].copy()
    recs["similarity_score"] = scores[filtered]
    return recs, scores[filtered]


def recommend(query_path: str):
    """Modo recomendação: recebe caminho de imagem e exibe similares."""
    print(f"\n=== RECOMENDAÇÃO ===")
    print(f"Imagem de consulta: {query_path}")

    query_path = Path(query_path)
    if not query_path.exists():
        print(f"[ERRO] Imagem não encontrada: {query_path}")
        sys.exit(1)

    # Cria a pasta imagem_busca se não existir e copia a imagem
    pasta_busca = BASE_DIR / "imagem_busca"
    pasta_busca.mkdir(parents=True, exist_ok=True)
    caminho_copia = pasta_busca / query_path.name
    try:
        shutil.copy2(query_path, caminho_copia)
        print(f"Cópia da imagem salva em: {caminho_copia}")
        query_path = caminho_copia
    except Exception as e:
        print(f"[AVISO] Não foi possível salvar cópia em imagem_busca: {e}")

    embeddings, valid_df = load_index()
    pca = load_pca()
    model, preprocess_fn = load_model(MODEL_BACKBONE)

    recs, scores = get_recommendations(
        query_path, embeddings, valid_df, model, preprocess_fn, pca)

    print(f"\nTop {TOP_N} produtos mais similares:")
    print("-" * 60)
    for _, row in recs.iterrows():
        name = row.get("productDisplayName", row["id"])
        cat = row.get("articleType", "?")
        score = row["similarity_score"]
        print(f"  [{score:.3f}]  {name}  ({cat})")

    # Salva imagem com resultado
    output_file = RESULTS_DIR / f"recomendacao_{query_path.stem}.png"
    save_recommendation_figure(query_path, recs, scores, output_file)
    print(f"\nFigura salva em: {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def save_recommendation_figure(query_path, recs, scores, output_file):
    n = len(recs)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 3, 3.5))

    # Imagem de consulta
    axes[0].imshow(Image.open(query_path))
    axes[0].set_title("CONSULTA", fontsize=9,
                      fontweight="bold", color="#1a56db")
    axes[0].axis("off")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#1a56db")
        spine.set_linewidth(2)
        spine.set_visible(True)

    # Recomendações
    for ax, (_, row), score in zip(axes[1:], recs.iterrows(), scores):
        path = row["image_path"]
        try:
            ax.imshow(Image.open(path))
        except Exception:
            ax.text(0.5, 0.5, "sem imagem", ha="center",
                    va="center", transform=ax.transAxes)
        name = str(row.get("productDisplayName", row["id"]))[:28]
        ax.set_title(f"{name}\n{score:.3f}", fontsize=7)
        ax.axis("off")

    plt.suptitle("Sistema de Recomendação por Imagem",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def evaluate():
    """Avalia a precisão do sistema por categoria (% mesma categoria)."""
    print("\n=== AVALIAÇÃO ===")
    embeddings, valid_df = load_index()
    pca = load_pca()
    model, preprocess_fn = load_model(MODEL_BACKBONE)

    top_cats = valid_df["articleType"].value_counts().head(10).index
    results = {}
    N_SAMPLES = 20

    for cat in top_cats:
        subset = valid_df[valid_df["articleType"] == cat]
        if len(subset) < 5:
            continue

        samples = subset.sample(min(N_SAMPLES, len(subset)), random_state=42)
        correct = 0
        total = 0

        for _, row in tqdm(samples.iterrows(), desc=cat, total=len(samples), leave=False):
            try:
                recs, _ = get_recommendations(
                    row["image_path"], embeddings, valid_df, model, preprocess_fn, pca, top_n=TOP_N
                )
                correct += (recs["articleType"] == cat).sum()
                total += TOP_N
            except Exception:
                pass

        if total > 0:
            results[cat] = correct / total
            print(f"  {cat:<25} {results[cat]:.1%}")

    if results:
        media = sum(results.values()) / len(results)
        print(f"\nPrecisão média: {media:.1%}")

        # Salva gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        cats = list(results.keys())
        vals = [results[c] for c in cats]
        ax.barh(cats, vals, color="#2563eb")
        ax.axvline(media, color="red", linestyle="--",
                   label=f"Média: {media:.1%}")
        ax.set_xlabel("Precisão (% mesma categoria)")
        ax.set_title("Avaliação por Categoria")
        ax.legend()
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        plt.tight_layout()
        out = RESULTS_DIR / "avaliacao_categorias.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gráfico salvo em: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# GERENCIAMENTO DO ÍNDICE (add / remove)
# ─────────────────────────────────────────────────────────────────────────────

def add_product(image_path: str, product_id: str):
    """Adiciona um único produto ao índice existente sem reconstruir tudo."""
    print(f"\n=== ADICIONANDO PRODUTO {product_id} ===")
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[ERRO] Imagem não encontrada: {image_path}")
        sys.exit(1)

    embeddings, valid_df = load_index()
    pca = load_pca()
    model, preprocess_fn = load_model(MODEL_BACKBONE)

    # Verifica se ID já existe
    if str(product_id) in valid_df["id"].astype(str).values:
        print(
            f"[AVISO] Produto {product_id} já existe no índice. Use --mode remove primeiro.")
        sys.exit(1)

    arr = preprocess_image(image_path, preprocess_fn)
    feat = normalize(model.predict(arr, verbose=0))
    if pca is not None:
        feat = normalize(pca.transform(feat))

    embeddings = np.vstack([embeddings, feat])

    new_row = pd.DataFrame([{
        "id":         product_id,
        "image_path": str(image_path),
    }])
    valid_df = pd.concat([valid_df, new_row], ignore_index=True)

    # Persiste
    np.save(EMBEDDINGS_FILE, embeddings)
    valid_df.drop(columns=["image_path"], errors="ignore").to_csv(
        PRODUCTS_FILE, index=False)
    print(f"Produto {product_id} adicionado. Total: {len(valid_df)} produtos.")


def remove_product(product_id: str):
    """Remove um produto do índice pelo ID."""
    print(f"\n=== REMOVENDO PRODUTO {product_id} ===")
    embeddings, valid_df = load_index()

    mask = valid_df["id"].astype(str) == str(product_id)
    if not mask.any():
        print(f"[ERRO] Produto {product_id} não encontrado no índice.")
        sys.exit(1)

    idx = valid_df.index[mask].tolist()
    embeddings = np.delete(embeddings, idx, axis=0)
    valid_df = valid_df.drop(index=idx).reset_index(drop=True)

    np.save(EMBEDDINGS_FILE, embeddings)
    valid_df.drop(columns=["image_path"], errors="ignore").to_csv(
        PRODUCTS_FILE, index=False)
    print(f"Produto {product_id} removido. Total: {len(valid_df)} produtos.")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO: testa com uma imagem aleatória do índice
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """Pega um produto aleatório do índice e mostra recomendações."""
    print("\n=== DEMO ===")
    embeddings, valid_df = load_index()
    pca = load_pca()
    model, preprocess_fn = load_model(MODEL_BACKBONE)

    sample = valid_df.sample(1).iloc[0]
    print(
        f"Produto sorteado: {sample.get('productDisplayName', sample['id'])}")
    print(f"Categoria: {sample.get('articleType', '?')}")

    recs, scores = get_recommendations(
        sample["image_path"], embeddings, valid_df, model, preprocess_fn, pca
    )

    print(f"\nTop {TOP_N} similares:")
    for _, row in recs.iterrows():
        print(
            f"  [{row['similarity_score']:.3f}]  {row.get('productDisplayName', row['id'])}")

    out = RESULTS_DIR / f"demo_{sample['id']}.png"
    save_recommendation_figure(sample["image_path"], recs, scores, out)
    print(f"\nFigura salva em: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendação por Similaridade de Imagens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--mode",
        choices=["build", "recommend", "evaluate", "add", "remove", "demo"],
        required=True,
        help="Modo de operação"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Caminho da imagem (obrigatório nos modos 'recommend' e 'add')"
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="ID do produto (obrigatório nos modos 'add' e 'remove')"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_N,
        help=f"Número de recomendações (padrão: {TOP_N})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global TOP_N
    TOP_N = args.top

    # Sempre verifica se o dataset local existe antes de prosseguir
    verificar_dataset()

    if args.mode == "build":
        build_index()

    elif args.mode == "recommend":
        if not args.image:
            print("[ERRO] Informe --image para o modo recommend")
            sys.exit(1)
        recommend(args.image)

    elif args.mode == "evaluate":
        evaluate()

    elif args.mode == "add":
        if not args.image or not args.id:
            print("[ERRO] Informe --image e --id para o modo add")
            sys.exit(1)
        add_product(args.image, args.id)

    elif args.mode == "remove":
        if not args.id:
            print("[ERRO] Informe --id para o modo remove")
            sys.exit(1)
        remove_product(args.id)

    elif args.mode == "demo":
        demo()


if __name__ == "__main__":
    main()
