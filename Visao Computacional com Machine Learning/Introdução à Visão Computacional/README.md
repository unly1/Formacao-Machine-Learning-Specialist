# Introdução à Visão Computacional

Este repositório contém as anotações e os conceitos fundamentais apresentados nas aulas teóricas do módulo de **Introdução à Visão Computacional**.

---

## 📚 Cronograma e Conteúdo das Aulas

### 🎥 Aula 1 - Introdução à Visão Computacional
Esta aula aborda os fundamentos de como os computadores capturam e interpretam o mundo visual, bem como os componentes básicos que integram um sistema de visão computacional.

*   **Sensoriamento:** O processo físico e digital de captura da luz através de sensores (matrizes de fotossensores) e sua conversão em dados de pixels.
*   **Processamento de Imagem:** Técnicas de pré-processamento como filtragem, suavização, realce de contraste e detecção de bordas para preparar os dados visuais.
*   **Análise com Machine Learning (ML):** Como utilizar algoritmos de aprendizado de máquina para analisar padrões e extrair características significativas a partir de representações numéricas de imagens.
*   **Reconhecimento de Pessoas:** Exemplos práticos e conceituais de detecção e identificação de indivíduos, ilustrados por referências da cultura pop (como a interface de HUD e identificação do filme *Robocop*).
*   **Posicionamento da Câmera:** Fatores cruciais na aquisição de dados, incluindo ângulo de visão, altura, calibração de lentes, profundidade de campo e iluminação do ambiente.

---

### 🧠 Aula 2 - Algoritmos de Deep Learning
Estudo da revolução provocada pelas Redes Neurais Convolucionais (CNNs) e pelas técnicas de aprendizagem profunda no processamento de imagens e vídeos.

*   **Aprendizado com Deep Learning:** Redes neurais artificiais multicamadas que aprendem representações hierárquicas e extração automática de características diretamente dos dados brutos.
*   **Detecção de Imagens:** Aplicação prática de modelos para inferir quais categorias de objetos estão presentes nos dados visuais.
*   **Como Criar um Dataset:**
    *   Estratégias para coleta de dados representativos e diversos.
    *   Definição de boas práticas para balanceamento de classes e tratamento de ruídos.
*   **Rotulação de Objetos (Labeling):** O processo de anotação de dados (criação de *bounding boxes*, máscaras de segmentação e rótulos textuais) essencial para o treinamento supervisionado de redes neurais.

---

### ⚙️ Aula 3 - Algoritmos de Visão Computacional
Exploração aprofundada dos diferentes níveis de análise de imagem e de como ir além da classificação estática para entender a dinâmica de cenas reais.

*   **Detecção em Imagens:** Localização de um ou mais objetos de interesse por meio de coordenadas espaciais.
*   **Rastreamento (Tracking):** Técnica de acompanhamento contínuo de objetos específicos em sequências de vídeo (frames consecutivos), permitindo inferir trajetórias e manter a identidade dos objetos ao longo do tempo (além da simples classificação/detecção frame a frame).
*   **Tipos de Tarefas de Análise Visual:**
    *   **Classification (Classificação):** Determina a classe principal presente na imagem (*"Existe um cachorro nesta foto?"*).
    *   **Classification + Localization (Classificação + Localização):** Determina a classe e desenha uma caixa delimitadora (*bounding box*) ao redor do objeto principal.
    *   **Object Detection (Detecção de Objetos):** Localiza e classifica múltiplos objetos de diferentes classes na mesma cena, desenhando caixas delimitadoras para cada um deles.
    *   **Instance Segmentation (Segmentação de Instâncias):** Detecta e delimita individualmente cada objeto mapeando os pixels exatos pertencentes a cada instância específica (diferenciando, por exemplo, múltiplos pedestres uns dos outros).
