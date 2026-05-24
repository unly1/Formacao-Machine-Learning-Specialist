# 🎭 Sistema de Reconhecimento Facial Inteligente

Projeto completo de **detecção e reconhecimento facial** equipado com uma **Interface Web Interativa (Streamlit Dashboard)** de alta fidelidade e dois backends de inteligência artificial independentes.

| Backend | Tecnologia | Melhor para |
|---|---|---|
| **face_recognition** | dlib HOG + SVM/KNN | Processamento veloz, sem necessidade de GPU, setup simples |
| **TensorFlow** | MobileNetV2 (Transfer Learning) | Datasets maiores, aprendizado profundo, GPU disponível |

> **Status:** 🟢 Testado, corrigido e 100% funcional no Windows 11 com Python 3.12.3

---

## 📁 Estrutura do Projeto Atualizada

A estrutura do projeto foi simplificada, eliminando redundâncias e centralizando a inicialização na raiz:

```text
Desafio - Criando um sistema de reconhecimento facial do zero/
│
├── README.md                         # Este manual do projeto
├── app.py                            # ← INTERFACE WEB INTERATIVA (Ponto de Entrada Principal)
│
└── facial_recognition/               # Diretório contendo a lógica do projeto
    ├── main.py                       # Orquestrador CLI (Linha de Comando)
    ├── requirements.txt              # Dependências do projeto (incluindo Streamlit)
    │
    ├── src/
    │   ├── download_dataset.py       # Baixa e prepara a base de imagens LFW
    │   ├── train_tensorflow.py       # Treina MobileNetV2 (TensorFlow/Keras)
    │   ├── train_face_recognition.py # Treina KNN/SVM nos embeddings extraídos da dlib
    │   ├── detect_images.py          # Detecção e reconhecimento em imagens estáticas
    │   └── detect_webcam.py          # Detecção e reconhecimento em tempo real (webcam)
    │
    ├── dataset/
    │   ├── train/                    # Imagens de treino organizadas por pasta/pessoa
    │   └── test/                     # Imagens de teste organizadas por pasta/pessoa
    │
    ├── models/
    │   ├── face_recognition_tf.keras # Modelo de classificação da rede convolucional (TF)
    │   ├── face_encodings.pkl        # Arquivo de embeddings (dlib) + classificadores SVM e KNN
    │   └── class_indices.json        # Mapeamento de classe numérica para nome (TF)
    │
    └── outputs/
        ├── detections/               # Saídas das fotos analisadas
        ├── screenshots/              # Capturas salvas durante o feed da câmera
        ├── comparison_fr.png         # Gráfico comparativo dlib (KNN vs SVM)
        └── training_curves.png       # Gráfico de curvas do treino do TensorFlow
```

---

## ⚙️ Instalação (Windows)

### 1. Acesse o ambiente virtual do workspace (Recomendado)
Este projeto usa a pasta `.venv` na raiz do seu workspace. Para ativá-lo no PowerShell do VS Code:
```powershell
.venv\Scripts\activate
```

### 2. Instale as dependências base
```powershell
pip install tensorflow>=2.12.0 opencv-python scikit-learn matplotlib Pillow tqdm requests scipy imutils
```

### 3. Instale a biblioteca dlib (Precompilada para Windows)
> ⚠️ **A instalação padrão do dlib via `pip install dlib` falha no Windows** por exigir o compilador C++ do Visual Studio instalado localmente. Use o wheel pré-compilado seguro:

```powershell
# Baixe o arquivo binário compilado:
Invoke-WebRequest -Uri "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl" -OutFile "dlib.whl" -UseBasicParsing

# Instale no ambiente virtual:
pip install dlib.whl

# Exclua o arquivo baixado temporário:
Remove-Item dlib.whl
```

### 4. Ajuste a compatibilidade numpy + dlib (Mandatório)
O wheel do dlib é incompatível com numpy 2.x. Faça o downgrade:
```powershell
pip install "numpy<2.0"
```
*Isso corrige o erro de imagem sem suporte (`RuntimeError: Unsupported image type`).*

### 5. Instale o face-recognition e o Streamlit
```powershell
pip install face-recognition streamlit
```

---

## 🚀 Como Usar a Interface Web (Recomendado)

A interface gráfica é o canal principal do projeto. Ela une todas as etapas do pipeline de forma amigável no seu navegador.

### Iniciar o Dashboard
Na pasta raiz do desafio (`Desafio - Criando um sistema de reconhecimento facial do zero`), execute:
```powershell
& '..\.venv\Scripts\streamlit' run app.py
```
*O sistema abrirá automaticamente uma aba no seu navegador padrão.*
Acesse manualmente em: **[http://localhost:8501](http://localhost:8501)**

### Recursos no Dashboard:
1. **Aba "Dataset & Cadastro"**:
   * Veja estatísticas gerais e consulte um banco de fotos em grid dos rostos já cadastrados.
   * **Cadastro de novo rosto**: Digite um nome e clique em "Iniciar Captura". O Streamlit abrirá sua webcam e tirará 20 fotos automáticas estruturando os diretórios de treino e teste sem que você precise fazer isso na mão.
2. **Aba "Treinamento dos Modelos"**:
   * Treine o modelo Dlib (SVM/KNN) acompanhando uma barra de carregamento visual passo a passo ou o TensorFlow com o console de treinamento de épocas integrado. Veja os resultados na hora com os gráficos comparativos.
   * Baixe também o dataset público oficial **LFW (Labeled Faces in the Wild)** com um único clique.
3. **Aba "Reconhecimento Facial (Webcam)"**:
   * Ligue a webcam direto no navegador, escolha qual inteligência artificial utilizar por trás, controle o limiar de aceitação (confiança) por um slider em tempo real e teste o reconhecimento.

---

## 💻 Como Usar via Linha de Comando (CLI)

Se preferir o terminal de comando, entre na pasta `facial_recognition`:
```powershell
cd facial_recognition
```

### 1. Baixar e preparar o Dataset
```powershell
$env:PYTHONUTF8=1; python main.py --mode download
```

### 2. Treinar os Modelos
```powershell
# Treinar face_recognition (SVM/KNN):
$env:PYTHONUTF8=1; python main.py --mode train_fr

# Treinar TensorFlow:
$env:PYTHONUTF8=1; python main.py --mode train_tf
```

### 3. Testar em Imagens Estáticas
```powershell
$env:PYTHONUTF8=1; python main.py --mode image --input foto.jpg --backend fr --show
```

### 4. Testar via Webcam Direta (Janela OpenCV)
```powershell
$env:PYTHONUTF8=1; python main.py --mode webcam --backend fr --classifier svm
```

---

## 📊 Acurácia Registrada
Resultados finais de precisão do sistema sob o dataset clássico LFW (96 pessoas distintas):

*   **SVM (Kernel RBF, C=10)**: **99.34%** Acurácia 🏆
*   **KNN (k=3, Euclidean)**: **99.07%** Acurácia

---

## 📌 Referências

*   Base LFW: [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
*   Projeto base dlib wheels: [z-mahmud22/Dlib_Windows_Python3.x](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)
*   Biblioteca Face Recognition: [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
