# 🎭 Lógica e Módulos do Sistema de Reconhecimento Facial

Este diretório contém o código-fonte, datasets e modelos compilados do sistema de reconhecimento facial. 

A interface gráfica web (**Streamlit Dashboard**) está localizada um nível acima na pasta raiz (`../app.py`), enquanto os scripts CLI principais residem neste diretório.

---

## 📁 Estrutura deste Diretório

```text
facial_recognition/ (Este Diretório)
│
├── main.py                       # Orquestrador principal em linha de comando (CLI)
├── requirements.txt              # Arquivo de dependências atualizado (inclui o Streamlit)
├── README.md                     # Este manual de desenvolvedor
│
├── src/                          # Scripts de execução das etapas do projeto
│   ├── download_dataset.py       # Baixa e separa a base de imagens LFW
│   ├── train_tensorflow.py       # Treina MobileNetV2 (TensorFlow/Keras)
│   ├── train_face_recognition.py # Treina KNN/SVM nos embeddings da dlib
│   ├── detect_images.py          # Detecção e reconhecimento em imagens
│   └── detect_webcam.py          # Detecção e reconhecimento na webcam via OpenCV
│
├── dataset/                      # Banco de imagens para treino/teste
│   ├── train/                    # 15 fotos de treino por pessoa
│   └── test/                     # 5 fotos de teste por pessoa
│
├── models/                       # Arquivos dos modelos de IA exportados
│   ├── face_recognition_tf.keras # Modelo convolucional treinado (TF)
│   ├── face_encodings.pkl        # Embeddings + classificadores SVM e KNN (dlib)
│   └── class_indices.json        # Mapeamento do classificador TensorFlow
│
└── outputs/                      # Diretório de relatórios, gráficos e prints
    ├── detections/               # Imagens salvas com rostos identificados
    ├── screenshots/              # Screenshots tiradas durante o feed da webcam
    ├── comparison_fr.png         # Gráfico de acurácia (SVM vs KNN)
    └── training_curves.png       # Curvas de treino do TensorFlow
```

---

## ⚙️ Instalação (A partir desta pasta)

Certifique-se de estar com seu ambiente virtual ativado:
```powershell
# Ativar o .venv do workspace:
..\.venv\Scripts\activate
```

Instale as dependências:
```powershell
pip install -r requirements.txt
```

> ⚠️ **Atenção:** O `dlib` necessita de um wheel pré-compilado para rodar no Windows sem erros de CMake:
> ```powershell
> Invoke-WebRequest -Uri "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl" -OutFile "dlib.whl" -UseBasicParsing
> pip install dlib.whl
> Remove-Item dlib.whl
> pip install "numpy<2.0"
> ```

---

## 🚀 Como Executar

### 🖥️ 1. Pela Interface Gráfica (Recomendado)
Para abrir o Dashboard visual a partir desta pasta:
```powershell
& '..\..\.venv\Scripts\streamlit' run ../app.py
```
Acesse em: **[http://localhost:8501](http://localhost:8501)**

---

### 💻 2. Pela Linha de Comando (CLI)

#### Pipeline Completo (Download + Treinos + Webcam)
```powershell
$env:PYTHONUTF8=1; python main.py --mode all
```

#### Passo a Passo Individual:

1. **Baixar dataset LFW**:
   ```powershell
   $env:PYTHONUTF8=1; python main.py --mode download
   ```
2. **Treinar apenas dlib (SVM/KNN)**:
   ```powershell
   $env:PYTHONUTF8=1; python main.py --mode train_fr
   ```
3. **Treinar apenas TensorFlow (MobileNetV2)**:
   ```powershell
   $env:PYTHONUTF8=1; python main.py --mode train_tf
   ```
4. **Reconhecimento em Imagem estática**:
   ```powershell
   $env:PYTHONUTF8=1; python main.py --mode image --input caminho_da_foto.jpg --show
   ```
5. **Reconhecimento na Webcam (Janela OpenCV)**:
   ```powershell
   $env:PYTHONUTF8=1; python main.py --mode webcam
   ```

---

## 📊 Performance Registrada no Dataset LFW
*   **SVM (Kernel RBF, C=10)**: **99.34%** Acurácia
*   **KNN (k=3, Euclidean)**: **99.07%** Acurácia
