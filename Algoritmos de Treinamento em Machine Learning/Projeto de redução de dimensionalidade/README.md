# 🖼️ Projeto de Redução de Dimensionalidade (Binarização de Imagens)

Este projeto faz parte da **Formação Machine Learning Specialist** e demonstra como aplicar a redução de dimensionalidade em dados de imagem de forma prática. O objetivo é converter uma imagem colorida (canal RGB tridimensional) para escala de cinza (canal único unidimensional) e, por fim, binarizar a imagem em preto e branco (representação binária de 1 bit).

Toda a lógica de conversão de cor, cálculo de estatísticas, plotagem de histograma em texto e o algoritmo de limiarização automática (Otsu) são implementados **em Python puro**, sem o uso de bibliotecas de visão computacional (como OpenCV ou Scikit-Image) para as operações matemáticas. O `Pillow` é utilizado exclusivamente para carregar os formatos modernos de imagem (como PNG/JPG) de forma a facilitar a execução prática.

---

## 📈 Etapas da Redução de Dimensionalidade

### 1. Escala de Cinza (Gray Scale)
Para reduzir a dimensão tridimensional RGB $(R, G, B)$ para um único canal de luminância, aplicamos a fórmula padrão da recomendação **ITU-R BT.601**:

$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

Esta fórmula pondera cada canal de cor com base na sensibilidade fisiológica do olho humano ao verde, vermelho e azul, respectivamente.

### 2. Binarização (Limiarização)
A binarização mapeia cada tom de cinza $Y$ para preto ($0$) ou branco ($255$) baseado em um limiar ($T$):

$$\text{Pixel}(x, y) = \begin{cases} 255 & \text{se } Y(x, y) \ge T \\ 0 & \text{se } Y(x, y) < T \end{cases}$$

O script suporta duas formas de definir o limiar:
- **Manual**: Um valor inteiro entre $0$ e $255$ fornecido pelo usuário.
- **Automático (Método de Otsu)**: Calcula o limiar ótimo maximizando a variância entre as duas classes de pixels (fundo e primeiro plano), resultando na melhor segmentação possível sem intervenção humana.

---

## 🛠️ Pré-requisitos

Para rodar o script utilizando imagens modernas como `.png` ou `.jpg`, certifique-se de ativar o ambiente virtual do projeto, que já contém o `Pillow` instalado.

```powershell
# Ativar o ambiente virtual a partir da raiz do repositório
.venv\Scripts\Activate.ps1
```

---

## 💻 Como Executar

O script suporta execução automática utilizando a imagem padrão `Sonic.png` ou a passagem manual de qualquer imagem PPM, PNG, JPG, JPEG ou BMP.

### 1. Execução Padrão (Fácil)
Basta rodar o script sem argumentos. Ele detectará automaticamente a imagem do Sonic no diretório `imagem/`, calculará o limiar via Otsu e salvará os resultados.

```powershell
python "Algoritmos de Treinamento em Machine Learning/Projeto de redução de dimensionalidade/binarizacao_pura.py"
```

### 2. Execução com Imagem Personalizada e Threshold Manual
Você pode passar o caminho da imagem de entrada e o valor do threshold desejado (ex: `128`):

```powershell
python "Algoritmos de Treinamento em Machine Learning/Projeto de redução de dimensionalidade/binarizacao_pura.py" caminho/para/foto.png 128
```

### 3. Execução com Imagem Personalizada e Limiar Automático (Otsu)
Passe a palavra-chave `auto` no segundo argumento para forçar o algoritmo de Otsu na sua imagem:

```powershell
python "Algoritmos de Treinamento em Machine Learning/Projeto de redução de dimensionalidade/binarizacao_pura.py" caminho/para/foto.png auto
```

---

## 📊 Outputs Gerados

Os resultados do processamento são criados na mesma pasta da imagem de entrada:
1. **`[nome]_cinza.[extensão]`**: Imagem reduzida para escala de cinza (canal de luminância).
2. **`[nome]_binaria.[extensão]`**: Imagem final binarizada (preto e branco).
3. **Estatísticas no Console**: Exibição das dimensões da imagem, desvio padrão, média, mediana e um **histograma gerado diretamente em ASCII** no terminal para visualização imediata da distribuição de brilho.
