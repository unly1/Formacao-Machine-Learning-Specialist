# Sistema de Recomendação por Similaridade de Imagens

Projeto desenvolvido como parte da Formação Machine Learning Specialist — Processamento de Imagens com Machine Learning.

O sistema recebe uma imagem de produto (relógio, camiseta, sapato etc.) e retorna os **N produtos mais similares visualmente**, sem depender de dados textuais, preço ou marca — apenas da aparência física do item.

---

## Como funciona

```
Imagem de entrada (Selecionada ou Externa)
      ↓
Pré-processamento (resize 224×224, normalização)
      ↓
ResNet50 — Transfer Learning (pesos ImageNet)
      ↓
Vetor de Embedding (2048 dimensões)
      ↓
PCA opcional (comprime para redução de dimensionalidade)
      ↓
Cosine Similarity contra o banco de embeddings
      ↓
Top-N produtos recomendados mostrados na interface
```

---

## Estrutura do Projeto

```
Desafio - Criando um Sistema de Recomendação por Imagens Digitais/
│
├── interface_recomendacao.py       ← Interface gráfica desktop (Tkinter)
├── sistema_recomendacao_imagens.py ← Engine principal (CLI & Pipeline)
│
├── archive/                         ← Pasta do dataset
│   ├── images/                     ← Imagens originais do dataset (.jpg)
│   └── styles.csv                  ← Metadados e categorização dos produtos
│
├── imagem_busca/                   ← Cópias das imagens selecionadas para consulta
├── cache/                          ← Embeddings brutos salvos (gerado automaticamente)
├── modelo_recomendacao/            ← Artefatos do índice treinado e persistido
│   ├── embeddings.npy              ← Banco de vetores dos produtos
│   ├── produtos.csv                ← Listagem dos produtos indexados
│   ├── pca.pkl                     ← Modelo PCA ajustado
│   └── config.json                 ← Metadados de configuração do modelo
│
└── resultados/                     ← Figuras e gráficos de recomendação salvos (CLI & GUI)
```

---

## Requisitos e Instalação

O projeto foi configurado para rodar utilizando o ambiente virtual Python localizado na raiz do repositório (`.venv`).

### Ativação do ambiente (.venv) no Windows (PowerShell):
```powershell
# Ativar o ambiente virtual
..\..\..\.venv\Scripts\activate
```

### Principais dependências instaladas:
- `tensorflow` (Extrator ResNet50)
- `scikit-learn` (PCA e Cossine Similarity)
- `pandas` e `numpy` (Manipulação de dados e vetores)
- `Pillow` (Processamento de imagem)
- `matplotlib` e `tqdm` (Visualizações e progresso no terminal)

---

## Preparação do Dataset

O dataset utilizado é o **Fashion Product Images (Small)**, disponível para download no Kaggle no link:
👉 [Fashion Product Images (Small) - Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small/data)

Como a base de dados deve ser baixada e configurada localmente, certifique-se de extraí-la dentro da pasta do projeto com a seguinte estrutura:
- As imagens do produto em: `archive/images/` (contendo os arquivos `.jpg`)
- O arquivo de metadados em: `archive/styles.csv`

---

## Interface Gráfica Interativa (GUI)

O projeto conta com uma interface desktop completa construída em **Tkinter** com design escuro premium (*Dark Theme*). 

Para abrir a interface, execute:
```bash
python interface_recomendacao.py
```

### Funcionalidades da Interface:
- **Painel Lateral**:
  - Seleção de imagem a partir do dataset (usando imagens presentes na pasta `archive/images`).
  - Botão de busca para imagens externas (qualquer arquivo `.jpg`/`.png` no seu computador).
  - Controle de quantidade de recomendações (`Top-N`).
  - Indicador de status de carregamento do TensorFlow e do índice.
- **Grid de Recomendações**:
  - Exibição em cartões com a foto do produto similar.
  - Exibição da porcentagem de similaridade, ID, marca, cor e gênero.
  - Indicador de barra colorida para representar visualmente a similaridade.

---

## Execução via Linha de Comando (CLI)

O script principal `sistema_recomendacao_imagens.py` também pode ser controlado via argumentos:

### 1. Construir o índice
Gera os embeddings e treina o PCA salvando os arquivos em `modelo_recomendacao/`:
```bash
python sistema_recomendacao_imagens.py --mode build
```

### 2. Rodar uma recomendação rápida (Modo Demo)
Escolhe uma imagem aleatória do dataset e mostra suas recomendações:
```bash
python sistema_recomendacao_imagens.py --mode demo
```

### 3. Recomendar a partir de uma imagem específica
```bash
python sistema_recomendacao_imagens.py --mode recommend --image archive\images\1163.jpg
```
Também funciona com qualquer imagem de fora do dataset:
```bash
python sistema_recomendacao_imagens.py --mode recommend --image C:\Users\usuario\Downloads\foto.jpg
```

### 4. Avaliar a precisão das categorias
Calcula a taxa de acerto do recomendador considerando se as imagens recomendadas pertencem à mesma categoria que a imagem de consulta:
```bash
python sistema_recomendacao_imagens.py --mode evaluate
```

### 5. Adicionar ou remover produtos manualmente
```bash
# Adicionar
python sistema_recomendacao_imagens.py --mode add --image nova_imagem.jpg --id 999999

# Remover
python sistema_recomendacao_imagens.py --mode remove --id 12345
```

---

## Configurações Principais

No início do arquivo `sistema_recomendacao_imagens.py`, você pode ajustar parâmetros como:

- `MAX_IMAGES`: Limite de imagens processadas no índice.
  - Atualmente configurado em **`50`** para fins de desenvolvimento e testes rápidos.
  - Para indexar o dataset completo, altere o valor para **`None`** ou para um número maior (ex: `5000`).
- `MODEL_BACKBONE`: O modelo de rede neural a ser utilizado. Opções: `"resnet50"` (padrão) ou `"vgg16"`.
- `TOP_N`: Número padrão de recomendações a exibir.
