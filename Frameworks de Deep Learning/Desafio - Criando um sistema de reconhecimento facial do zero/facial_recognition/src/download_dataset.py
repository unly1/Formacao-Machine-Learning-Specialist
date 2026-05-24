"""
download_dataset.py
-------------------
Baixa e prepara o dataset LFW (Labeled Faces in the Wild) — dataset público de faces.
Seleciona pessoas com pelo menos MIN_IMAGES imagens para garantir treino adequado.
"""

import os
import urllib.request
import tarfile
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

# ── Configurações ─────────────────────────────────────────────────────────────
# Mirrors do LFW em ordem de preferência
LFW_MIRRORS = [
    "https://github.com/brendan-lewis/lfw-dataset/releases/download/v1.0/lfw.tgz",
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
]
RAW_DIR       = Path("dataset/raw")
PROCESSED_DIR = Path("dataset/processed")
TRAIN_DIR     = Path("dataset/train")
TEST_DIR      = Path("dataset/test")

MIN_IMAGES    = 15      # mínimo de imagens por pessoa para incluir no dataset
IMG_SIZE      = (160, 160)
TRAIN_RATIO   = 0.80
RANDOM_SEED   = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class DownloadProgress(tqdm):
    """Barra de progresso para download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_via_sklearn(lfw_dir: Path) -> bool:
    """
    Tenta baixar o LFW via scikit-learn (fetch_lfw_people).
    Reconstrói a estrutura de pastas por pessoa em dataset/raw/lfw/.
    Retorna True se bem-sucedido.
    """
    try:
        from sklearn.datasets import fetch_lfw_people
        print("[i] Baixando LFW via scikit-learn (fetch_lfw_people)...")
        print("    (Isso pode demorar alguns minutos na primeira vez)")
        dataset = fetch_lfw_people(
            min_faces_per_person=MIN_IMAGES,
            resize=None,
            color=True,
            download_if_missing=True,
        )
        from sklearn.datasets import _base as sk_base
        import inspect
        # Localiza onde o sklearn salvou os dados
        data_home = Path(sk_base.get_data_home())
        sklearn_lfw = data_home / "lfw_home" / "lfw_funneled"
        if not sklearn_lfw.exists():
            # tenta caminho alternativo
            sklearn_lfw = data_home / "lfw_home" / "lfw"

        if sklearn_lfw.exists():
            print(f"[i] Copiando de {sklearn_lfw} para {lfw_dir} ...")
            if lfw_dir.exists():
                shutil.rmtree(lfw_dir)
            shutil.copytree(sklearn_lfw, lfw_dir)
            print(f"[OK] Dataset copiado para: {lfw_dir}")
            return True
        else:
            print(f"[!] Pasta sklearn nao encontrada em: {sklearn_lfw}")
            return False
    except Exception as e:
        print(f"[!] Falha ao baixar via sklearn: {e}")
        return False


def download_lfw():
    """Faz o download do LFW tentando múltiplas fontes."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tgz_path = RAW_DIR / "lfw.tgz"
    lfw_dir  = RAW_DIR / "lfw"

    if lfw_dir.exists():
        print(f"[OK] Dataset LFW ja encontrado em: {lfw_dir}")
        return lfw_dir

    # Tenta cada mirror de URL
    for url in LFW_MIRRORS:
        try:
            print(f"[v] Tentando baixar de: {url}")
            with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc="lfw.tgz") as t:
                urllib.request.urlretrieve(url, tgz_path, reporthook=t.update_to)

            print("[^] Extraindo arquivo ...")
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(RAW_DIR)

            if tgz_path.exists():
                tgz_path.unlink()  # remove o .tgz para economizar espaço

            if lfw_dir.exists():
                print(f"[OK] Extraido em: {lfw_dir}")
                return lfw_dir
        except Exception as e:
            print(f"[!] Falha no mirror {url}: {e}")
            if tgz_path.exists():
                tgz_path.unlink()

    # Fallback: sklearn
    print("[i] Todos os mirrors falharam. Tentando via scikit-learn...")
    if download_via_sklearn(lfw_dir):
        return lfw_dir

    raise RuntimeError(
        "Nao foi possivel baixar o dataset LFW.\n"
        "Por favor, baixe manualmente em: http://vis-www.cs.umass.edu/lfw/lfw.tgz\n"
        "e extraia em: dataset/raw/lfw/"
    )


def preprocess_image(src_path: Path, dst_path: Path):
    """Lê, redimensiona e salva uma imagem em escala de cinza."""
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    img = cv2.resize(img, IMG_SIZE)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), img)
    return True


def filter_and_split(lfw_dir: Path):
    """
    Seleciona pessoas com >= MIN_IMAGES fotos,
    pré-processa e divide em treino/teste.
    """
    # Limpa diretórios anteriores
    for d in (PROCESSED_DIR, TRAIN_DIR, TEST_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    people = sorted([p for p in lfw_dir.iterdir() if p.is_dir()])
    qualified = [p for p in people if len(list(p.glob("*.jpg"))) >= MIN_IMAGES]

    print(f"\n[i] Total de pessoas no LFW : {len(people)}")
    print(f"[i] Pessoas com ≥ {MIN_IMAGES} imagens: {len(qualified)}")
    print(f"[i] Divisão treino/teste    : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}\n")

    label_map = {}
    for label_id, person_dir in enumerate(tqdm(qualified, desc="Pré-processando")):
        name   = person_dir.name
        images = sorted(person_dir.glob("*.jpg"))
        random.shuffle(images)

        split       = int(len(images) * TRAIN_RATIO)
        train_imgs  = images[:split]
        test_imgs   = images[split:]

        label_map[label_id] = name

        for img_path in train_imgs:
            dst = TRAIN_DIR / name / img_path.name
            preprocess_image(img_path, dst)

        for img_path in test_imgs:
            dst = TEST_DIR / name / img_path.name
            preprocess_image(img_path, dst)

    # Salva o mapeamento label → nome
    import json
    with open("dataset/label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Dataset pronto!")
    print(f"    Treino : {TRAIN_DIR}")
    print(f"    Teste  : {TEST_DIR}")
    print(f"    Labels : dataset/label_map.json")
    return label_map


def main():
    print("=" * 55)
    print("  Sistema de Reconhecimento Facial — Preparação LFW")
    print("=" * 55)
    lfw_dir   = download_lfw()
    label_map = filter_and_split(lfw_dir)
    print(f"\n[i] Classes disponíveis ({len(label_map)}):")
    for lid, name in list(label_map.items())[:10]:
        print(f"    {lid:3d}. {name}")
    if len(label_map) > 10:
        print(f"    … e mais {len(label_map)-10} pessoas.")


if __name__ == "__main__":
    main()
