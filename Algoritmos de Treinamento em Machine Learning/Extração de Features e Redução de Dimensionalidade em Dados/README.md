# Extração de Features e Redução de Dimensionalidade em Dados

Este projeto faz parte da **Formação Machine Learning Specialist** e demonstra a aplicação prática de diversas técnicas de engenharia de atributos (feature engineering), redução de dimensionalidade (linear e não-linear) e seleção de atributos (feature selection).

## 🚀 Técnicas Demonstradas no Script

### 1. Engenharia / Extração de Features
* **PolynomialFeatures**: Geração de features polinomiais e interações cruzadas automáticas.
* **Interações Manuais**: Criação manual de produtos par-a-par entre variáveis.
* **Estatísticas de Janela (Rolling)**: Geração de médias móveis, desvios padrões, valores mínimos/máximos acumulados e lags de atraso temporal para séries temporais.

### 2. Redução de Dimensionalidade Linear
* **PCA (Principal Component Analysis)**: Redução de dimensões maximizando a variância explicada.
* **LDA (Linear Discriminant Analysis)**: Redução de dimensionalidade supervisionada focada na separabilidade de classes.
* **TruncatedSVD**: Fatoração de matrizes esparsas.
* **ICA (Independent Component Analysis)**: Separação de sinais em componentes estatisticamente independentes.

### 3. Redução de Dimensionalidade Não-Linear (Manifold Learning)
* **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Mapeamento probabilístico não-linear de alta fidelidade para agrupamentos.
* **Isomap**: Redução não-linear preservando as distâncias geodésicas.
* **LLE (Locally Linear Embedding)**: Preservação de simetrias locais dos vizinhos mais próximos.
* **UMAP** (Opcional se `umap-learn` estiver instalado).

### 4. Seleção de Features
* **VarianceThreshold**: Remoção de variáveis com variância abaixo de um limite pré-estabelecido.
* **Correlação Alta**: Filtro manual para eliminar redundâncias com correlação superior a 0.85.
* **SelectKBest (F-score e Mutual Information)**: Seleção univariada das melhores features para classificação.
* **RFE (Recursive Feature Elimination)**: Eliminação recursiva de atributos guiada pela importância gerada por uma Random Forest.
* **RFECV**: RFE integrado com Validação Cruzada para identificar o número ideal de atributos.

---

## 🛠️ Pré-requisitos

Para rodar o script localmente, certifique-se de ter o Python instalado e as seguintes bibliotecas no seu ambiente virtual:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn
```

---

## 💻 Como Executar Localmente

Como o projeto possui um ambiente virtual pré-configurado na raiz (`.venv`), você pode seguir os passos abaixo para executar no Windows usando o **PowerShell** ou **CMD**:

### Utilizando o PowerShell (Recomendado)

1. Abra o terminal na pasta raiz do repositório (`Formacao-Machine-Learning-Specialist`).
2. Ative o ambiente virtual:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
3. Navegue até a pasta do módulo ou execute o script diretamente apontando o caminho:
   ```powershell
   python "Algoritmos de Treinamento em Machine Learning/Extração de Features e Redução de Dimensionalidade em Dados/features_dimensionalidade.py"
   ```

### Executando diretamente sem ativação (PowerShell / CMD)

Se preferir rodar o script diretamente chamando o executável do Python dentro do ambiente virtual `.venv`, execute a partir do diretório raiz:

```powershell
& ".venv/Scripts/python.exe" "Algoritmos de Treinamento em Machine Learning/Extração de Features e Redução de Dimensionalidade em Dados/features_dimensionalidade.py"
```

---

## 📊 Resultados e Outputs Gerados

Ao rodar o script com sucesso, os seguintes artefatos são gerados na mesma pasta do script:

1. **`resultados_features_dimensionalidade.txt`**:
   Arquivo contendo toda a saída textual impressa no console, incluindo o log passo a passo e a tabela final com o resumo comparativo das dimensões de entrada e saída para cada técnica.

2. **Diretório `outputs/`**:
   Pasta contendo os 6 gráficos comparativos das transformações no formato `.png`:
   * `pca_variancia.png`: Gráfico da curva de variância explicada acumulada e por componente.
   * `reducao_linear.png`: Comparativo 2D dos resultados do PCA, LDA, TruncatedSVD e ICA.
   * `manifold.png`: Comparativo 2D das técnicas de Manifold Learning (t-SNE, Isomap, LLE e UMAP se disponível).
   * `feature_selection_scores.png`: Visualização em barras dos scores de importância por F-score e Mutual Information.
   * `rfecv.png`: Curva de desempenho (Acurácia) de acordo com a quantidade de variáveis selecionadas via validação cruzada.
   * `correlacao.png`: Mapa de calor de correlação entre os atributos sintéticos redundantes.
