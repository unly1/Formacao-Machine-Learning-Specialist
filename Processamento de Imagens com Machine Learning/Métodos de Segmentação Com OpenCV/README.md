# 🧠 Métodos de Segmentação de Imagens (OpenCV & Deep Learning)

Este diretório reúne o conteúdo teórico e prático sobre **Métodos de Segmentação de Imagens**, cobrindo desde abordagens clássicas baseadas em processamento digital de imagens (como detecção de bordas com o filtro de Sobel e Haar Cascades/DNN para detecção facial) até abordagens modernas de segmentação semântica e de instância com Deep Learning (Mask R-CNN e SegNet).

---

## 📅 Estrutura das Aulas e Projetos

### 📖 Aula 1 - Métodos de Segmentação com OpenCV
Estudo de métodos clássicos de segmentação baseados em descontinuidades locais da imagem (bordas). O foco reside no **Operador de Sobel**, que calcula o gradiente bidimensional da intensidade de tons de cinza.

* **Conceitos Teóricos:**
  * **Gradiente de Imagem:** Mudanças rápidas de intensidade nos pixels que delimitam a transição entre objetos.
  * **Kernels de Sobel ($3 \times 3$):**
    * $G_x$: Detecta variações horizontais (bordas verticais).
    * $G_y$: Detecta variações verticais (bordas horizontais).
  * **Magnitude do Gradiente:** A força total da borda calculada pela combinação dos dois eixos: $G = \sqrt{G_x^2 + G_y^2}$.
* **Aplicação Prática:**
  * O script [exemplo.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Processamento%20de%20Imagens%20com%20Machine%20Learning/M%C3%A9todos%20de%20Segmenta%C3%A7%C3%A3o%20Com%20OpenCV/Metodos%20de%20Segmenta%C3%A7%C3%A3o%20com%20OpenCV/exemplo.py) implementa a detecção de bordas utilizando a imagem clássica da Lenna (ou cria uma imagem sintética como fallback).
  * Ele salva as bordas processadas no eixo X, no eixo Y e a magnitude total das bordas na pasta de resultados, gerando também um painel comparativo geral.

---

### 📖 Aula 2 - Projetos com OpenCV
Nesta etapa, exploramos a integração prática de modelos de Deep Learning otimizados no OpenCV através do seu módulo de rede neural profunda (`cv2.dnn`).

* **Conceitos Teóricos:**
  * **Módulo `cv2.dnn`:** Interface eficiente do OpenCV para carregar e rodar inferências em modelos pré-treinados de outros frameworks (como Caffe, TensorFlow e PyTorch).
  * **SSD (Single Shot MultiBox Detector) com ResNet-10:** Arquitetura leve e rápida de detecção de objetos, ideal para processamento em tempo real.
  * **Captura Dinâmica:** Uso de `cv2.VideoCapture` para ler frames de vídeo diretamente da webcam do usuário.
* **Aplicação Prática:**
  * O script [face_detection.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Processamento%20de%20Imagens%20com%20Machine%20Learning/M%C3%A9todos%20de%20Segmenta%C3%A7%C3%A3o%20Com%20OpenCV/Projetos%20com%20OpenCV/face_detection.py) gerencia a webcam para capturar uma foto com a tecla `ESPAÇO`.
  * Ele carrega os arquivos do modelo (`deploy.prototxt` e `res10_300x300_ssd_iter_140000.caffemodel`), realiza a inferência para detectar faces e desenha caixas delimitadoras com o percentual de confiança de cada detecção.

---

### 📖 Aula 3 - Métodos de Segmentação Semântica
Introdução teórica às arquiteturas modernas de segmentação de pixels por meio de redes convolucionais profundas (CNNs). Diferenciamos a **Segmentação Semântica** (onde agrupamos pixels da mesma classe sem identificar indivíduos únicos) da **Segmentação de Instância** (onde cada unidade individual de uma classe é identificada separadamente).

* **Modelos Estudados:**
  1. **Mask R-CNN (Region-based CNN):**
     * **Foco:** Segmentação de Instância (*Instance Segmentation*).
     * **Funcionamento:** Baseado na Faster R-CNN (para caixas delimitadoras), adiciona uma ramificação paralela para prever uma máscara de segmentação pixel a pixel (FCN - *Fully Convolutional Network*) para cada Região de Interesse (RoI).
     * **RoIAlign:** Substitui o antigo RoIPool para extrair mapas de características alinhados espacialmente sem perda de precisão geométrica por arredondamento, garantindo máscaras de alta resolução espacial.
  2. **SegNet (referenciada como "signet"):**
     * **Foco:** Segmentação Semântica (*Semantic Segmentation*).
     * **Arquitetura Encoder-Decoder:**
       * **Encoder:** Camadas convolucionais (geralmente baseadas na VGG-16) reduzem a resolução e extraem características geométricas robustas da imagem.
       * **Decoder:** Reconstrói a imagem ao tamanho original para efetuar a classificação binária ou multiclasse pixel a pixel.
     * **Índices de Max Pooling:** A principal inovação da SegNet é salvar a posição espacial (índice) do elemento máximo em cada pooling do encoder e usá-la no decoder para realizar o upsampling de forma direta, sem a necessidade de aprender parâmetros extras para isso, reduzindo drasticamente o consumo de memória.

---

### 📖 Aula 4 - Métodos de Segmentação Semântica na prática
Exercício prático aplicando o pipeline de segmentação semântica do modelo **DeepLabv3** (Google) em imagens reais.

* **Funcionamento do Pipeline:**
  1. **Model Loader:** Download e carregamento do modelo pré-treinado **DeepLabv3** (com base em MobileNetV2 ou Xception) treinado no dataset PASCAL VOC.
  2. **Inference:** Processamento da imagem de entrada redimensionada (tamanho máximo de 513) e inferência clássica via Grafo do TensorFlow para obter as predições de classes por pixel.
  3. **Visualização e Resultados:**
     * Mapeamento de cores baseado na paleta PASCAL VOC para colorir as diferentes classes detectadas (como pessoas, carros, planos de fundo).
     * Exibição de um painel comparativo contendo a imagem original, o mapa de segmentação e o overlay translúcido com legenda das classes únicas encontradas.
     * Armazenamento automático do painel de resultado dentro do diretório local `resultados/`.
* **Script Prático Associado:**
  * O código-fonte atualizado e parametrizável pode ser encontrado no script [deeplab.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Processamento%20de%20Imagens%20com%20Machine%20Learning/M%C3%A9todos%20de%20Segmenta%C3%A7%C3%A3o%20Com%20OpenCV/M%C3%A9todos%20de%20Segmenta%C3%A7%C3%A3o%20Sem%C3%A2ntica%20na%20pr%C3%A1tica/deeplab.py) dentro do diretório [Métodos de Segmentação Semântica na prática](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Processamento%20de%20Imagens%20com%20Machine%20Learning/M%C3%A9todos%20de%20Segmenta%C3%A7%C3%A3o%20Com%20OpenCV/M%C3%A9todos%20de%20Segmenta%C3%A7%C3%A3o%20Sem%C3%A2ntica%20na%20pr%C3%A1tica).
