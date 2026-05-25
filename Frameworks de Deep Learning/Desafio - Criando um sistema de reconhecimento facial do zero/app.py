from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import pickle
import json
import time
import os
import sys

# Garante que o diretório de trabalho seja o da pasta onde app.py está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Muda o diretório de trabalho para facial_recognition caso executado na raiz do desafio
if os.path.exists("facial_recognition"):
    os.chdir("facial_recognition")
sys.path.append(os.getcwd())


# Configuração da Página
st.set_page_config(
    page_title="Reconhecimento Facial IA",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeção de CSS customizado para Visual Premium e Moderno
st.markdown("""
<style>
    /* Estilo Sci-Fi/Dark para o Dashboard */
    .stApp {
        background-color: #0E1117;
        color: #E2E8F0;
    }
    
    /* Título Principal */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(45deg, #00F2FE, #4FACFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 2px solid #1E293B;
        padding-bottom: 10px;
        margin-bottom: 25px;
    }
    
    /* Cards com efeito de Glassmorphism */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Indicador Tech */
    .tech-badge {
        background-color: #1E293B;
        border-left: 4px solid #00F2FE;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        font-family: monospace;
        color: #00F2FE;
        font-size: 0.9em;
        margin-bottom: 15px;
    }

    /* Estilo de Botões */
    .stButton>button {
        background: linear-gradient(135deg, #0052D4, #4364F7, #6FB1FC);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(67, 100, 247, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(67, 100, 247, 0.6);
        background: linear-gradient(135deg, #0052D4, #4364F7, #6FB1FC);
    }
</style>
""", unsafe_allow_html=True)

# ── Configurações de Caminho ──────────────────────────────────────────────────
TRAIN_DIR = Path("dataset/train")
TEST_DIR = Path("dataset/test")
MODEL_DIR = Path("models")
FR_ENCODINGS = MODEL_DIR / "face_encodings.pkl"
TF_MODEL_PATH = MODEL_DIR / "face_recognition_tf.keras"
CLASS_IDX_PATH = MODEL_DIR / "class_indices.json"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper: Carregar Reconhecedores Dinamicamente ──────────────────────────────


def get_fr_classes():
    if FR_ENCODINGS.exists():
        with open(FR_ENCODINGS, "rb") as f:
            data = pickle.load(f)
            return list(data["label_encoder"].classes_)
    return []


def get_tf_classes():
    if CLASS_IDX_PATH.exists():
        with open(CLASS_IDX_PATH) as f:
            idx_to_class = json.load(f)
            return list(idx_to_class.values())
    return []


# ── Menu Lateral ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/face-id.png", width=80)
    st.markdown("### Facial Recognition System")
    st.markdown(
        "Dashboard de controle para administração do dataset, treinamento inteligente e detecção via webcam.")
    st.markdown("---")
    st.markdown("**Status do Sistema:**")

    # Indicadores rápidos de modelos
    fr_ok = FR_ENCODINGS.exists()
    tf_ok = TF_MODEL_PATH.exists()

    st.markdown(
        f"• **dlib/face_recognition:** {'🟢 Treinado' if fr_ok else '🔴 Não Treinado'}")
    st.markdown(
        f"• **TensorFlow (CNN):** {'🟢 Treinado' if tf_ok else '🔴 Não Treinado'}")

    st.markdown("---")
    st.caption("Desenvolvido para Formação Machine Learning Specialist")

# ── Título do App ─────────────────────────────────────────────────────────────
st.title("👤 Sistema de Reconhecimento Facial IA")

# Criação das Abas
tab_data, tab_train, tab_webcam = st.tabs([
    "📊 Dataset & Cadastro",
    "⚙️ Treinamento dos Modelos",
    "🎥 Reconhecimento Facial (Webcam)"
])

# ==============================================================================
# ABA 1: DATASET & CADASTRO
# ==============================================================================
with tab_data:
    st.subheader("Gerenciamento do Dataset")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Estatísticas do Dataset")

        # Estatísticas rápidas das classes
        people = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
        total_people = len(people)

        total_train_imgs = sum(len(list(d.glob("*.jpg")))
                               for d in TRAIN_DIR.iterdir() if d.is_dir())
        total_test_imgs = sum(len(list(d.glob("*.jpg")))
                              for d in TEST_DIR.iterdir() if d.is_dir())

        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("Pessoas Cadastradas", total_people)
        m_c2.metric("Imagens de Treino", total_train_imgs)
        m_c3.metric("Imagens de Teste", total_test_imgs)

        if total_people > 0:
            st.markdown("#### Pessoas e número de fotos (Treino):")
            # Lista com contagem de fotos
            stats_list = []
            for p in people:
                count = len(list((TRAIN_DIR / p).glob("*.jpg")))
                stats_list.append({"Pessoa": p, "Imagens": count})
            st.dataframe(stats_list, use_container_width=True, height=200)
        else:
            st.warning(
                "O dataset está vazio. Baixe a base LFW na aba de Treinamento ou cadastre novos rostos.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📸 Cadastrar Novo Rosto via Webcam")
        st.markdown(
            "Digite o nome da pessoa abaixo e tire 20 fotos para registrá-la.")

        reg_name = st.text_input(
            "Nome Completo (sem caracteres especiais):", key="reg_name")
        start_capture = st.button("Iniciar Captura de Fotos")

        if start_capture:
            if not reg_name:
                st.error("Por favor, preencha o nome antes de capturar.")
            else:
                name_clean = "".join(
                    [c for c in reg_name if c.isalnum() or c in (" ", "_", "-")]).strip()
                if not name_clean:
                    st.error("Nome inválido! Utilize apenas letras e números.")
                else:
                    train_p = TRAIN_DIR / name_clean
                    test_p = TEST_DIR / name_clean
                    train_p.mkdir(parents=True, exist_ok=True)
                    test_p.mkdir(parents=True, exist_ok=True)

                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("Não foi possível acessar a webcam local.")
                    else:
                        st.info(
                            "Câmera ligada! Olhe para a webcam e mude ligeiramente a expressão facial.")
                        time.sleep(1)

                        placeholder = st.empty()
                        pbar = st.progress(0)
                        status_msg = st.empty()

                        count = 0
                        while count < 20:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Erro ao ler frame da câmera.")
                                break

                            # Mostra o feed ao usuário
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            placeholder.image(
                                rgb_frame, width=320, channels="RGB")

                            # Salva 15 no treino, 5 no teste
                            if count < 15:
                                filename = train_p / f"face_{count}.jpg"
                            else:
                                filename = test_p / f"face_{count-15}.jpg"

                            cv2.imwrite(str(filename), frame)
                            count += 1
                            pbar.progress(count / 20)
                            status_msg.write(f"Capturado: {count}/20 fotos...")
                            time.sleep(0.15)  # intervalo curto entre fotos

                        cap.release()
                        placeholder.empty()
                        pbar.empty()
                        status_msg.empty()
                        st.success(
                            f"✓ '{name_clean}' cadastrado com sucesso! (15 fotos de treino, 5 de teste)")
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Galeria de Imagens
    if total_people > 0:
        st.markdown("### 🖼️ Galeria do Dataset")
        selected_person = st.selectbox(
            "Escolha uma pessoa para ver as fotos:", people)

        person_dir = TRAIN_DIR / selected_person
        if person_dir.exists():
            img_paths = list(person_dir.glob("*.jpg")
                             )[:12]  # Limita a 12 imagens
            if img_paths:
                cols = st.columns(6)
                for idx, path in enumerate(img_paths):
                    col_idx = idx % 6
                    with cols[col_idx]:
                        st.image(str(path), use_container_width=True,
                                 caption=f"Foto {idx+1}")
            else:
                st.info("Nenhuma imagem de treino encontrada para esta pessoa.")

# ==============================================================================
# ABA 2: TREINAMENTO DOS MODELOS
# ==============================================================================
with tab_train:
    st.subheader("Treinamento de Inteligência Artificial")

    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Treinamento Rápido (dlib/face_recognition)")
        st.markdown("Este método extrai as características faciais (embeddings de 128 dimensões) usando Dlib e treina classificadores clássicos rápidos (SVM / KNN). Recomendado para computadores comuns sem GPU dedicada.")

        train_fr_btn = st.button("Iniciar Treinamento dlib (SVM/KNN)")

        if train_fr_btn:
            if total_people < 2:
                st.error(
                    "É necessário ter pelo menos 2 pessoas cadastradas no dataset para treinar os classificadores!")
            else:
                st.info(
                    "Iniciando pipeline de extração e treinamento... Acompanhe abaixo:")

                # Elementos visuais de progresso
                fr_progress_bar = st.progress(0)
                fr_status = st.empty()

                # Callback de progresso
                def streamlit_fr_cb(i, total, name, split):
                    pct = (i + 1) / total
                    fr_progress_bar.progress(pct)
                    fr_status.text(
                        f"[{split.upper()}] Processando: {name} ({i+1}/{total})")

                # Importa e roda
                try:
                    from src.train_face_recognition import train_pipeline
                    with st.spinner("Treinando classificadores SVM e KNN nos dados extraídos..."):
                        results = train_pipeline(progress_cb=streamlit_fr_cb)

                    if results:
                        st.success(
                            f"✓ Treinamento finalizado! Melhor Modelo: {results['best']} (SVM: {results['svm_acc']*100:.1f}% | KNN: {results['knn_acc']*100:.1f}%)")

                        # Mostra o gráfico de comparação gerado
                        comparison_img = Path("outputs/comparison_fr.png")
                        if comparison_img.exists():
                            st.image(
                                str(comparison_img), caption="Resultados dlib (KNN vs SVM)", width=500)
                    else:
                        st.error("Erro durante a execução do treinamento.")
                except Exception as e:
                    st.error(f"Ocorreu um erro no treinamento: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        st.markdown('</div>', unsafe_allow_html=True)

        # Pipeline Completa / Botão Download
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📥 Dataset Oficial (LFW)")
        st.markdown(
            "Se você quiser baixar e extrair o dataset de celebridades LFW (Labeled Faces in the Wild) para popular o sistema:")

        download_btn = st.button("Baixar Dataset LFW Oficial")
        if download_btn:
            st.info(
                "Baixando base LFW... Isso pode levar de 1 a 2 minutos dependendo da sua internet.")
            try:
                from src.download_dataset import download_and_extract_lfw, split_dataset
                # Roda o download
                download_and_extract_lfw()
                split_dataset()
                st.success(
                    "✓ Dataset LFW baixado, processado e distribuído em dataset/train e dataset/test!")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao baixar: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_t2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Treinamento TensorFlow (CNN MobileNetV2)")
        st.markdown("Treina uma Rede Neural Convolucional profunda no TensorFlow. Este processo realiza Transfer Learning e pode levar alguns minutos dependendo do seu hardware.")

        epochs_head = st.number_input(
            "Épocas iniciais (Camadas Superiores):", min_value=1, max_value=50, value=10)
        epochs_fine = st.number_input(
            "Épocas de Fine-Tuning:", min_value=0, max_value=50, value=15)

        train_tf_btn = st.button("Iniciar Treinamento TensorFlow")

        if train_tf_btn:
            if total_people < 2:
                st.error(
                    "É necessário ter pelo menos 2 pessoas cadastradas no dataset para treinar!")
            else:
                st.info(
                    "Treinamento TensorFlow iniciado. O log em tempo real aparecerá abaixo:")

                # Executa o script via subprocesso e captura os logs do console em tempo real
                log_placeholder = st.empty()

                import subprocess
                import sys

                cmd = [
                    sys.executable,
                    "src/train_tensorflow.py",
                    "--epochs_head", str(epochs_head),
                    "--epochs_fine", str(epochs_fine)
                ]

                env = os.environ.copy()
                env["PYTHONUTF8"] = "1"

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env
                )

                full_log = ""
                # Lê a saída linha a linha
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        full_log += output
                        # Exibe as últimas 15 linhas do log para não sobrecarregar a tela
                        lines = full_log.split("\n")
                        log_placeholder.code("\n".join(lines[-15:]))

                rc = process.poll()
                if rc == 0:
                    st.success(
                        "✓ Treinamento TensorFlow concluído com sucesso!")

                    # Mostra os gráficos gerados pelo TF se existirem
                    metrics_img = Path("outputs/training_curves.png")
                    if metrics_img.exists():
                        st.image(str(
                            metrics_img), caption="Curvas de Treinamento e Validação - TensorFlow", use_container_width=True)
                else:
                    st.error(
                        f"Erro no treinamento do TensorFlow. Código de retorno: {rc}")
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# ABA 3: RECONHECIMENTO FACIAL (WEBCAM)
# ==============================================================================
with tab_webcam:
    st.subheader("Visualização e Inferência em Tempo Real")

    col_w1, col_w2 = st.columns([1, 2])

    with col_w1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Painel de Configuração")

        backend_opt = st.selectbox(
            "Backend de Detecção:",
            ["face_recognition (dlib)", "TensorFlow (CNN)"]
        )

        if backend_opt == "face_recognition (dlib)":
            selected_backend = "fr"
            clf_opt = st.selectbox("Classificador:", ["svm", "knn"])
        else:
            selected_backend = "tensorflow"
            clf_opt = "svm"

        conf_threshold = st.slider(
            "Limiar de Confiança:",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Limiar mínimo para identificar a pessoa. Se estiver abaixo, será classificado como desconhecido."
        )

        camera_id = st.number_input(
            "ID da Câmera (Normalmente 0):", min_value=0, max_value=5, value=0)

        # Botões de ativação
        st.markdown("---")
        start_webcam = st.toggle("🎥 Ativar Webcam", key="webcam_toggle")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_w2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📺 Transmissão de Vídeo")

        video_frame = st.empty()

        if start_webcam:
            # ── Inicialização do Recognizer selecionado ──
            recognizer = None
            try:
                if selected_backend == "tensorflow":
                    # Importação dinâmica das classes
                    from src.detect_webcam import TFRecognizer
                    if not TF_MODEL_PATH.exists() or not CLASS_IDX_PATH.exists():
                        st.error(
                            "O modelo do TensorFlow não está treinado! Vá para a aba 'Treinamento' primeiro.")
                        st.session_state["webcam_toggle"] = False
                        st.rerun()
                    recognizer = TFRecognizer()
                else:
                    from src.detect_webcam import FRRecognizer
                    if not FR_ENCODINGS.exists():
                        st.error(
                            "Os encodings dlib/face_recognition não estão treinados! Vá para a aba 'Treinamento' primeiro.")
                        st.session_state["webcam_toggle"] = False
                        st.rerun()
                    recognizer = FRRecognizer(classifier=clf_opt)
            except Exception as e:
                st.error(f"Erro ao carregar modelo: {e}")
                start_webcam = False

            if recognizer:
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    st.error(
                        f"Não foi possível abrir a câmera ID {camera_id}.")
                else:
                    # Configura câmera
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    fps_vals = []
                    t_prev = time.time()

                    st.info("Pressione o botão acima para desligar a webcam.")

                    # Loop de Captura e Processamento
                    while st.session_state.get("webcam_toggle", False):
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Falha na captura da câmera.")
                            break

                        # Processa reconhecimento
                        results = recognizer.process_frame(frame)

                        # Desenha os boxes e labels com a confiança ajustada do slider
                        for (x1, y1, x2, y2, name, conf) in results:
                            # Define cor do box de acordo com a confiança
                            color = (0, 180, 255) if conf >= conf_threshold else (
                                100, 100, 100)
                            label = f"{name} ({conf:.2f})" if conf >= conf_threshold else f"Desconhecido"

                            # Cantos decorativos estilo HUD
                            corner = 12
                            thick = 2
                            for (cx, cy, dx, dy) in [
                                (x1, y1, 1, 1), (x2, y1, -1, 1),
                                (x1, y2, 1, -1), (x2, y2, -1, -1)
                            ]:
                                cv2.line(frame, (cx, cy),
                                         (cx + dx*corner, cy), color, thick)
                                cv2.line(frame, (cx, cy),
                                         (cx, cy + dy*corner), color, thick)

                            # Retângulo translúcido interno
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (x1, y1),
                                          (x2, y2), color, -1)
                            cv2.addWeighted(
                                overlay, 0.05, frame, 0.95, 0, frame)

                            # Etiqueta com fundo
                            (tw, th), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                            cv2.rectangle(frame, (x1, y1 - th - 8),
                                          (x1 + tw + 6, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                        # Calcula FPS
                        t_now = time.time()
                        fps_vals.append(1.0 / max(t_now - t_prev, 1e-6))
                        t_prev = t_now
                        if len(fps_vals) > 15:
                            fps_vals.pop(0)
                        fps = np.mean(fps_vals)

                        # HUD
                        h, w = frame.shape[:2]
                        cv2.rectangle(frame, (0, 0), (w, 30), (15, 15, 15), -1)
                        info_text = f"[{backend_opt.split()[0].upper()}] | FPS: {fps:.1f} | Faces: {len(results)}"
                        cv2.putText(
                            frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 242, 254), 1, cv2.LINE_AA)

                        # Converte e plota
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_frame.image(
                            rgb_frame, channels="RGB", use_container_width=True)

                        # Pequeno delay para liberar a CPU
                        time.sleep(0.01)

                    cap.release()
                    video_frame.empty()
        else:
            st.info(
                "A webcam está desligada. Ative a webcam no menu ao lado para iniciar o reconhecimento facial em tempo real.")

        st.markdown('</div>', unsafe_allow_html=True)
