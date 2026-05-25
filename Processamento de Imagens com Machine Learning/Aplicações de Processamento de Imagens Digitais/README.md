# Aplicações de Processamento de Imagens Digitais

Este diretório reúne as anotações teóricas e práticas sobre as **Aplicações de Processamento de Imagens Digitais**, cobrindo desde filtragem espacial e morfologia matemática até métodos avançados de segmentação e teoria das cores.

---

## 📅 Conteúdo Programático por Aula

### 📖 Aula 1 - Aplicações de Processamento de Imagens Digitais
Nesta introdução, são abordados os métodos fundamentais para manipular e aprimorar imagens digitais, identificando seus diferentes estados e aplicando transformações geométricas e morfológicas.

* **Estados de Processamento da Imagem:**
  * **Original:** Imagem crua obtida pelo sensor.
  * **Com Ruído / Distorcida:** Imagem alterada por ruídos espaciais (ex: gaussiano, sal e pimenta) ou distorções óticas.
  * **Processada:** Imagem modificada para realce ou correção.
  * **Contraste:** Ampliação da faixa dinâmica de intensidades de pixel.
  * **Detecção de Bordas:** Extração de limites estruturais da imagem.
* **Filtros de Detecção de Bordas (Primeira Ordem):**
  * **Sobel:** Filtro que calcula o gradiente da intensidade da imagem, enfatizando transições de alta frequência espacial.
  * **Roberts & Prewitt:** Operadores alternativos para aproximação do gradiente de borda, variando o tamanho e os pesos das máscaras de convolução.
* **Morfologia Matemática:**
  * **Dilatação:** Expande as regiões brilhantes da imagem (adiciona pixels aos limites dos objetos).
  * **Erosão:** Reduz as regiões brilhantes da imagem (remove pixels nos limites dos objetos).
  * **Abertura (Erosão seguida de Dilatação):** Útil para remover pequenos ruídos ou isolar objetos conectados.
  * **Fechamento (Dilatação seguida de Erosão):** Útil para preencher pequenos buracos e fendas em estruturas.
  * **Exemplo Prático (Leitor de Impressão Digital):** Uso de abertura e fechamento para suavizar cristas papilares e remover imperfeições da captura digital.
  * **Transformada Hit-or-Miss (Acerto ou Erro):** Operador morfológico básico para detecção de formas ou padrões específicos com base na vizinhança de pixels de primeiro plano e de fundo.

---

### 📖 Aula 2 - Métodos de Segmentação
A segmentação consiste na partição de uma imagem em regiões homogêneas de interesse. Nesta aula, o foco reside na segmentação por bordas e descontinuidade de intensidade.

* **Segmentação por Detecção de Bordas:**
  * Baseada na detecção de descontinuidades locais de cinza ou cor.
  * Uso de operadores de gradiente bidimensional para mapear as fronteiras físicas dos objetos.
* **Operadores de Convolução:**
  * Comparativo prático e teórico entre os filtros **Sobel**, **Roberts** e **Prewitt** para estimativa do gradiente horizontal e vertical ($G_x$ e $G_y$).

---

### 📖 Aula 3 - Métodos Avançados de Segmentação
Abordagem de algoritmos robustos e modernos para isolamento de regiões complexas e sobrepostas.

* **Binarização (Limiarização / Thresholding):**
  * Divisão da imagem em primeiro plano e fundo usando limiares globais (ex: Método de Otsu) ou locais adaptativos.
* **Segmentação por Watershed (Bacia Hidrográfica):**
  * Metáfora geográfica que visualiza a imagem em tons de cinza como uma superfície topográfica. Linhas de crista (linhas divisórias de águas) delimitam as diferentes regiões, ideal para separar objetos sobrepostos.
* **Transformada Imagem-Floresta (IFT - Image Foresting Transform):**
  * Técnica baseada em grafos que modela a imagem como um grafo direcionado, onde a segmentação é resolvida por caminhos de custo mínimo a partir de marcadores (*seeds*).
* **Redes de Deep Learning:**
  * Segmentação semântica e de instância usando redes neurais profundas convolucionais (ex: arquiteturas **U-Net**, **Mask R-CNN** e **SegNet**), que aprendem representações hierárquicas diretamente dos pixels.

---

### 📖 Aula 4 - Métodos de Conversão de Cores
Estudo de como a luz e a cor são interpretadas pelo olho humano e processadas em ambientes computacionais.

* **A Fisiologia da Visão Humana:**
  * **Córnea e Íris:** Responsáveis pelo controle de entrada e refração da luz.
  * **Cristalino:** Lente natural que foca a luz na retina.
  * **Esclera:** Camada externa protetora.
  * **Retina:** Onde a imagem é formada e processada pelos fotorreceptores.
  * **Nervo Óptico:** Transmite os impulsos nervosos gerados pelos fotorreceptores ao cérebro.
  * **Cones e Bastonetes:** Cones são responsáveis pela visão cromática (cor, alta resolução) divididos em sensibilidades de vermelho, verde e azul. Bastonetes respondem à luz acromática (visão escotópica/noturna, alta sensibilidade de brilho).
* **Espectro Eletromagnético e Luz Visível:**
  * A luz visível é composta de radiação eletromagnética com comprimentos de onda variando de **400 nm** (violeta) até **700 nm** (vermelho).
* **Luz Acromática vs. Luz Cromática:**
  * **Acromática:** Luz sem cor (monocromática), caracterizada apenas pela intensidade ou brilho (tons de cinza).
  * **Cromática:** Luz contendo frequências de cores, caracterizada por três grandezas físicas:
    1. **Radiância:** Quantidade de energia total que flui de uma fonte luminosa (medida em Watts).
    2. **Luminância:** Quantidade de energia percebida pelo observador (medida em lumens).
    3. **Brilho:** Sensação subjetiva e não mensurável de intensidade luminosa do observador humano.
* **Espaços de Cores para Processamento:**
  * Representação matemática e espacial de cores em modelos computacionais como **RGB** (Red, Green, Blue), **HSV** (Hue, Saturation, Value) e **HSI** (Hue, Saturation, Intensity).

---

### 📖 Aula 5 - Formação de Cores em Processamento de Imagens
Estudo detalhado e matemático sobre os principais espaços de cores utilizados no desenvolvimento de sistemas de visão computacional.

* **Modelos e Espaços de Cores:**
  * **RGB:** Modelo aditivo baseado nas cores primárias vermelha, verde e azul, amplamente utilizado em displays.
  * **XYZ (CIE XYZ):** Espaço de cor padrão definido pela Comissão Internacional de Iluminação que modela a percepção de cor humana média.
  * **HSV (Hue, Saturation, Value) / HSL (Hue, Saturation, Lightness):** Modelos orientados à percepção humana, facilitando a segmentação baseada em matizes (*Hue*) e pureza da cor (*Saturation*).
