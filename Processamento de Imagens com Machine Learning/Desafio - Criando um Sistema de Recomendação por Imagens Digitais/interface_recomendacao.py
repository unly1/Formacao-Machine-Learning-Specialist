"""
Interface Desktop — Sistema de Recomendação por Imagens
========================================================
Execute com:
    python interface_recomendacao.py
"""

import os
import sys
import json
import pickle
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES (devem ser iguais ao sistema_recomendacao_imagens.py)
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "archive"        # pasta com images/ e styles.csv
IMAGES_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "styles.csv"
OUTPUT_DIR = BASE_DIR / "modelo_recomendacao"
RESULTS_DIR = BASE_DIR / "resultados"

EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
PRODUCTS_FILE = OUTPUT_DIR / "produtos.csv"
PCA_FILE = OUTPUT_DIR / "pca.pkl"
CONFIG_FILE = OUTPUT_DIR / "config.json"

IMG_SIZE = (224, 224)
TOP_N = 5

# Cores / tema
BG_DARK = "#0f0f1a"
BG_CARD = "#1a1a2e"
BG_PANEL = "#16213e"
ACCENT = "#7c3aed"       # roxo vibrante
ACCENT2 = "#06b6d4"       # ciano
TEXT_PRI = "#f1f5f9"
TEXT_SEC = "#94a3b8"
SUCCESS = "#10b981"
DANGER = "#ef4444"
BORDER = "#334155"

FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_SUB = ("Segoe UI", 11, "bold")
FONT_BODY = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)
FONT_BADGE = ("Segoe UI", 8, "bold")


# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA (ML) — carregamento lazy de TF
# ─────────────────────────────────────────────────────────────────────────────

def _load_tf():
    """Importa TensorFlow apenas quando necessário."""
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    return tf, ResNet50, preprocess_input


def load_index():
    if not EMBEDDINGS_FILE.exists() or not PRODUCTS_FILE.exists():
        raise FileNotFoundError(
            "Índice não encontrado.\n"
            "Rode primeiro:\n  python sistema_recomendacao_imagens.py --mode build"
        )
    embeddings = np.load(EMBEDDINGS_FILE)
    df = pd.read_csv(PRODUCTS_FILE)
    df["image_path"] = df["id"].apply(lambda x: IMAGES_DIR / f"{x}.jpg")
    return embeddings, df


def load_pca():
    if not PCA_FILE.exists():
        return None
    with open(PCA_FILE, "rb") as f:
        return pickle.load(f)


def load_config():
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE) as f:
        return json.load(f)


def preprocess_image_path(path, preprocess_fn):
    from tensorflow.keras.preprocessing import image as kimage
    img = kimage.load_img(str(path), target_size=IMG_SIZE)
    arr = kimage.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_fn(arr)


def get_recommendations(query_path, embeddings, df, model, preprocess_fn, pca, top_n):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    arr = preprocess_image_path(query_path, preprocess_fn)
    feat = normalize(model.predict(arr, verbose=0))
    if pca is not None:
        feat = normalize(pca.transform(feat))

    scores = cosine_similarity(feat, embeddings)[0]
    query_id = str(Path(query_path).stem)
    sorted_idx = np.argsort(scores)[::-1]
    filtered = [i for i in sorted_idx if str(
        df.iloc[i]["id"]) != query_id][:top_n]

    recs = df.iloc[filtered].copy()
    recs["similarity_score"] = scores[filtered]
    return recs


def save_recommendation_figure(query_path, recs, output_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n = len(recs)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 3, 3.5))

    # Imagem de consulta
    try:
        axes[0].imshow(Image.open(query_path))
    except Exception:
        axes[0].text(0.5, 0.5, "erro consulta", ha="center", va="center")
    axes[0].set_title("CONSULTA", fontsize=9, fontweight="bold", color="#1a56db")
    axes[0].axis("off")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#1a56db")
        spine.set_linewidth(2)
        spine.set_visible(True)

    # Recomendações
    for ax, (_, row) in zip(axes[1:], recs.iterrows()):
        path = row["image_path"]
        score = row.get("similarity_score", 0)
        try:
            ax.imshow(Image.open(path))
        except Exception:
            ax.text(0.5, 0.5, "sem imagem", ha="center", va="center")
        name = str(row.get("productDisplayName", row["id"]))[:28]
        ax.set_title(f"{name}\n{score:.3f}", fontsize=7)
        ax.axis("off")

    plt.suptitle("Sistema de Recomendação por Imagem", fontsize=11, fontweight="bold")
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE UI
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_tk(pil_img, size):
    pil_img = pil_img.resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


def make_rounded_button(parent, text, command, color=ACCENT, fg=TEXT_PRI,
                        font=FONT_SUB, padx=18, pady=8):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=color, fg=fg, font=font,
        relief="flat", cursor="hand2",
        activebackground=ACCENT2, activeforeground=TEXT_PRI,
        padx=padx, pady=pady, bd=0
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT2))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn


# ─────────────────────────────────────────────────────────────────────────────
# JANELA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class RecomendacaoApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("🔍 Sistema de Recomendação por Imagens")
        self.geometry("1280x820")
        self.minsize(1000, 700)
        self.configure(bg=BG_DARK)

        # Estado
        self._model = None
        self._preprocess = None
        self._embeddings = None
        self._df = None
        self._pca = None
        self._query_path = None
        self._tk_images = []    # mantém refs para GC
        self._top_n = tk.IntVar(value=TOP_N)

        self._build_ui()
        self._async_load_index()

    # ── Layout geral ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Barra superior
        self._build_header()

        # Corpo principal (sidebar | conteúdo)
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self._build_sidebar(body)
        self._build_main(body)

        # Barra de status
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_PANEL, height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Gradiente visual com label colorido
        title = tk.Label(
            hdr,
            text="🧠  Sistema de Recomendação por Imagens",
            bg=BG_PANEL, fg=TEXT_PRI,
            font=("Segoe UI", 15, "bold")
        )
        title.pack(side="left", padx=24, pady=16)

        subtitle = tk.Label(
            hdr,
            text="ResNet50 + Cosine Similarity",
            bg=BG_PANEL, fg=ACCENT2,
            font=("Segoe UI", 10)
        )
        subtitle.pack(side="left", pady=16)

        # Badge status do modelo
        self._model_badge = tk.Label(
            hdr,
            text="⏳  Carregando índice...",
            bg=BG_PANEL, fg=TEXT_SEC,
            font=FONT_BADGE
        )
        self._model_badge.pack(side="right", padx=24)

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=BG_CARD, width=280)
        side.pack(side="left", fill="y", padx=(0, 12), pady=0)
        side.pack_propagate(False)

        # ── Seção: imagem de consulta ──
        tk.Label(side, text="IMAGEM DE CONSULTA", bg=BG_CARD, fg=ACCENT,
                 font=FONT_BADGE).pack(anchor="w", padx=16, pady=(20, 6))

        # Área de drop / preview
        self._query_frame = tk.Frame(
            side, bg=BG_PANEL, width=248, height=248,
            relief="flat", bd=2
        )
        self._query_frame.pack(padx=16, pady=4)
        self._query_frame.pack_propagate(False)

        self._query_label = tk.Label(
            self._query_frame,
            text="Clique em\n'Selecionar Imagem'\npara começar",
            bg=BG_PANEL, fg=TEXT_SEC,
            font=FONT_BODY, justify="center"
        )
        self._query_label.pack(expand=True)

        # Botão selecionar
        btn_sel = make_rounded_button(
            side, "📂  Selecionar Imagem",
            self._select_image, color=ACCENT
        )
        btn_sel.pack(fill="x", padx=16, pady=(10, 4))

        # Botão recomendar
        self._btn_rec = make_rounded_button(
            side, "🔍  Buscar Similares",
            self._run_recommendation, color="#059669"
        )
        self._btn_rec.pack(fill="x", padx=16, pady=4)
        self._btn_rec.config(state="disabled")

        # ── Separador ──
        ttk.Separator(side, orient="horizontal").pack(
            fill="x", padx=16, pady=16)

        # ── Configurações ──
        tk.Label(side, text="CONFIGURAÇÕES", bg=BG_CARD, fg=ACCENT,
                 font=FONT_BADGE).pack(anchor="w", padx=16, pady=(0, 8))

        cfg_row = tk.Frame(side, bg=BG_CARD)
        cfg_row.pack(fill="x", padx=16)
        tk.Label(cfg_row, text="Top N resultados:", bg=BG_CARD,
                 fg=TEXT_SEC, font=FONT_BODY).pack(side="left")
        tk.Spinbox(
            cfg_row, from_=1, to=20, width=5,
            textvariable=self._top_n,
            bg=BG_PANEL, fg=TEXT_PRI,
            buttonbackground=BG_PANEL,
            relief="flat", font=FONT_BODY
        ).pack(side="right")

        ttk.Separator(side, orient="horizontal").pack(
            fill="x", padx=16, pady=16)

        # ── Info do índice ──
        tk.Label(side, text="INFO DO ÍNDICE", bg=BG_CARD, fg=ACCENT,
                 font=FONT_BADGE).pack(anchor="w", padx=16, pady=(0, 8))
        self._info_label = tk.Label(
            side,
            text="Aguardando carregamento...",
            bg=BG_CARD, fg=TEXT_SEC,
            font=FONT_MONO, justify="left", wraplength=240
        )
        self._info_label.pack(anchor="w", padx=16)



    def _build_main(self, parent):
        main = tk.Frame(parent, bg=BG_DARK)
        main.pack(side="left", fill="both", expand=True)

        # Título
        tk.Label(
            main, text="Resultados da Busca",
            bg=BG_DARK, fg=TEXT_PRI, font=FONT_TITLE
        ).pack(anchor="w", pady=(12, 4))

        tk.Label(
            main,
            text="Selecione uma imagem e clique em 'Buscar Similares' para ver produtos parecidos.",
            bg=BG_DARK, fg=TEXT_SEC, font=FONT_BODY
        ).pack(anchor="w", pady=(0, 12))

        # Frame de cards — scrollável
        canvas_frame = tk.Frame(main, bg=BG_DARK)
        canvas_frame.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(
            canvas_frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=self._canvas.yview)
        self._scrollable = tk.Frame(self._canvas, bg=BG_DARK)

        self._scrollable.bind(
            "<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all"))
        )
        self._canvas.create_window(
            (0, 0), window=self._scrollable, anchor="nw")
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mensagem placeholder
        self._placeholder = tk.Label(
            self._scrollable,
            text="🔍\n\nNenhuma busca realizada ainda.\nSelecione uma imagem para começar.",
            bg=BG_DARK, fg=TEXT_SEC,
            font=("Segoe UI", 13),
            justify="center"
        )
        self._placeholder.pack(expand=True, pady=80)

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=BG_PANEL, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_var = tk.StringVar(value="Pronto.")
        tk.Label(bar, textvariable=self._status_var,
                 bg=BG_PANEL, fg=TEXT_SEC, font=FONT_MONO).pack(side="left", padx=12)

        # Barra de progresso (oculta por padrão)
        self._progress = ttk.Progressbar(bar, mode="indeterminate", length=180)
        self._progress.pack(side="right", padx=12, pady=4)
        self._progress.pack_forget()

    # ── Carregamento assíncrono ───────────────────────────────────────────────

    def _async_load_index(self):
        self._set_status("Carregando índice de embeddings...", busy=True)
        threading.Thread(target=self._load_index_thread, daemon=True).start()

    def _load_index_thread(self):
        try:
            emb, df = load_index()
            pca = load_pca()
            cfg = load_config()
            self.after(0, lambda: self._on_index_loaded(emb, df, pca, cfg))
        except FileNotFoundError as e:
            self.after(0, lambda: self._on_index_error(str(e)))

    def _on_index_loaded(self, emb, df, pca, cfg):
        self._embeddings = emb
        self._df = df
        self._pca = pca

        n = cfg.get("n_products", len(df))
        bb = cfg.get("backbone", "resnet50").upper()
        dims = cfg.get("n_components_pca", emb.shape[1])

        self._info_label.config(
            text=f"✅ Produtos: {n:,}\n📐 Dims: {dims}D\n🧠 Backbone: {bb}",
            fg=SUCCESS
        )
        self._model_badge.config(
            text=f"✅  {n:,} produtos indexados", fg=SUCCESS)
        self._set_status(f"Índice carregado — {n:,} produtos.", busy=False)
        self._update_rec_button()

    def _on_index_error(self, msg):
        self._info_label.config(text=f"❌ {msg}", fg=DANGER)
        self._model_badge.config(text="❌  Índice não encontrado", fg=DANGER)
        self._set_status(
            "Índice não encontrado. Construa o índice primeiro.", busy=False)

    # ── Carregamento assíncrono do modelo TF ──────────────────────────────────

    def _async_load_model(self, callback):
        self._set_status(
            "Carregando ResNet50 (pode demorar na 1ª vez)...", busy=True)
        threading.Thread(target=self._load_model_thread,
                         args=(callback,), daemon=True).start()

    def _load_model_thread(self, callback):
        try:
            tf, ResNet50, preprocess_fn = _load_tf()
            model = ResNet50(weights="imagenet",
                             include_top=False, pooling="avg")
            model.trainable = False
            self.after(0, lambda: callback(model, preprocess_fn))
        except Exception as e:
            self.after(0, lambda: self._on_tf_error(str(e)))

    def _on_tf_error(self, msg):
        self._set_status(f"Erro ao carregar TensorFlow: {msg}", busy=False)
        messagebox.showerror(
            "TensorFlow não encontrado",
            "TensorFlow não está instalado.\n\n"
            "Instale com:\n  pip install tensorflow\n\n"
            f"Detalhe: {msg}"
        )

    # ── Seleção de imagem ─────────────────────────────────────────────────────

    def _select_image(self):
        path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")]
        )
        if not path:
            return

        import shutil
        pasta_busca = BASE_DIR / "imagem_busca"
        pasta_busca.mkdir(parents=True, exist_ok=True)
        caminho_copia = pasta_busca / Path(path).name
        try:
            shutil.copy2(path, caminho_copia)
            self._query_path = caminho_copia
            print(f"Cópia da imagem salva em: {caminho_copia}")
        except Exception as e:
            self._query_path = Path(path)
            print(f"Erro ao salvar cópia em imagem_busca: {e}")

        self._show_query_preview(self._query_path)
        self._set_status(f"Imagem selecionada: {self._query_path.name} (cópia em imagem_busca)")
        self._update_rec_button()

    def _show_query_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            tk_img = pil_to_tk(img, (240, 240))
            self._tk_images = [tk_img]   # limpa refs antigas

            self._query_label.config(image=tk_img, text="")
            self._query_label.image = tk_img
        except Exception as e:
            self._query_label.config(
                text=f"Erro ao abrir imagem:\n{e}", image="")

    # ── Recomendação ──────────────────────────────────────────────────────────

    def _update_rec_button(self):
        can = bool(self._query_path and self._embeddings is not None)
        self._btn_rec.config(state="normal" if can else "disabled")

    def _run_recommendation(self):
        if not self._query_path:
            return

        if self._model is None:
            self._async_load_model(self._on_model_ready)
        else:
            self._do_recommend()

    def _on_model_ready(self, model, preprocess_fn):
        self._model = model
        self._preprocess = preprocess_fn
        self._set_status("Modelo carregado.", busy=False)
        self._do_recommend()

    def _do_recommend(self):
        self._set_status("Buscando produtos similares...", busy=True)
        top_n = self._top_n.get()
        threading.Thread(target=self._recommend_thread,
                         args=(top_n,), daemon=True).start()

    def _recommend_thread(self, top_n):
        try:
            recs = get_recommendations(
                self._query_path,
                self._embeddings, self._df,
                self._model, self._preprocess,
                self._pca, top_n
            )
            
            # Salva o resultado na pasta resultados/
            output_file = RESULTS_DIR / f"recomendacao_{Path(self._query_path).stem}.png"
            try:
                save_recommendation_figure(self._query_path, recs, output_file)
                status_msg = f"{len(recs)} produto(s) similar(es) encontrado(s). Figura salva em resultados/."
            except Exception as fig_err:
                print(f"Erro ao salvar figura de recomendação: {fig_err}")
                status_msg = f"{len(recs)} produto(s) similar(es) encontrado(s)."
                
            self.after(0, lambda: self._show_results(recs, status_msg))
        except Exception as e:
            self.after(0, lambda: self._on_rec_error(str(e)))

    def _on_rec_error(self, msg):
        self._set_status(f"Erro: {msg}", busy=False)
        messagebox.showerror("Erro na recomendação", msg)

    # ── Exibição dos resultados ───────────────────────────────────────────────

    def _show_results(self, recs, status_msg):
        # Limpa área
        for w in self._scrollable.winfo_children():
            w.destroy()
        self._tk_images = [self._query_label.image] if hasattr(
            self._query_label, "image") else []

        self._set_status(status_msg, busy=False)

        # Grid de cards
        COLS = 3
        for i, (_, row) in enumerate(recs.iterrows()):
            col = i % COLS
            row_idx = i // COLS
            card = self._make_result_card(self._scrollable, row)
            card.grid(row=row_idx, column=col, padx=12, pady=12, sticky="nsew")

        for c in range(COLS):
            self._scrollable.columnconfigure(c, weight=1)

        self._canvas.yview_moveto(0)

    def _make_result_card(self, parent, row):
        card = tk.Frame(parent, bg=BG_CARD, relief="flat", bd=0)
        card.configure(padx=2, pady=2)

        # Hover
        def on_enter(e):
            card.config(bg=BG_PANEL)

        def on_leave(e):
            card.config(bg=BG_CARD)
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)

        # Imagem
        img_frame = tk.Frame(card, bg=BG_CARD, width=200, height=200)
        img_frame.pack(padx=12, pady=(12, 4))
        img_frame.pack_propagate(False)

        img_label = tk.Label(img_frame, bg=BG_CARD, text="📷",
                             fg=TEXT_SEC, font=("Segoe UI", 32))
        img_label.pack(expand=True)

        # Carrega imagem
        try:
            pil_img = Image.open(str(row["image_path"])).convert("RGB")
            tk_img = pil_to_tk(pil_img, (192, 192))
            self._tk_images.append(tk_img)
            img_label.config(image=tk_img, text="")
            img_label.image = tk_img
        except Exception:
            pass

        # Pontuação
        score = row.get("similarity_score", 0)
        score_pct = int(score * 100)
        score_color = SUCCESS if score_pct >= 80 else (
            ACCENT2 if score_pct >= 60 else TEXT_SEC)

        score_frame = tk.Frame(card, bg=BG_CARD)
        score_frame.pack(fill="x", padx=12, pady=2)

        tk.Label(
            score_frame, text="Similaridade",
            bg=BG_CARD, fg=TEXT_SEC, font=FONT_BADGE
        ).pack(side="left")

        # Barra de progresso visual
        bar_bg = tk.Frame(score_frame, bg=BORDER, height=8, width=80)
        bar_bg.pack(side="right", pady=4)
        bar_fill = tk.Frame(bar_bg, bg=score_color, height=8,
                            width=max(1, int(80 * score)))
        bar_fill.place(x=0, y=0)

        tk.Label(
            card, text=f"{score_pct}%",
            bg=BG_CARD, fg=score_color, font=("Segoe UI", 12, "bold")
        ).pack(pady=(0, 2))

        # Nome do produto
        name = str(row.get("productDisplayName", row.get("id", "—")))
        if len(name) > 40:
            name = name[:37] + "..."
        tk.Label(
            card, text=name,
            bg=BG_CARD, fg=TEXT_PRI,
            font=FONT_SUB, wraplength=200, justify="center"
        ).pack(padx=12, pady=2)

        # Metadados
        metas = []
        for col_name, label in [
            ("articleType",  "Tipo"),
            ("baseColour",   "Cor"),
            ("gender",       "Gênero"),
            ("season",       "Estação"),
        ]:
            val = row.get(col_name)
            if pd.notna(val) and str(val).strip():
                metas.append(f"{label}: {val}")

        if metas:
            tk.Label(
                card, text=" · ".join(metas[:2]),
                bg=BG_CARD, fg=TEXT_SEC,
                font=FONT_BADGE, wraplength=200
            ).pack(padx=12, pady=(0, 8))

        # ID
        tk.Label(
            card, text=f"ID: {row['id']}",
            bg=BG_CARD, fg=BORDER, font=FONT_MONO
        ).pack(pady=(0, 10))

        return card



    # ── Status / progresso ────────────────────────────────────────────────────

    def _set_status(self, msg, busy=None):
        self._status_var.set(msg)
        if busy is True:
            self._progress.pack(side="right", padx=12, pady=4)
            self._progress.start(10)
        elif busy is False:
            self._progress.stop()
            self._progress.pack_forget()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = RecomendacaoApp()
    app.mainloop()
