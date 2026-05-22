# 🧬 Treinamento Não-Supervisionado em Machine Learning — K-Means

Este diretório contém a implementação prática e o estudo do algoritmo de agrupamento **K-Means**, abordando desde conceitos teóricos (como o método de Expectativa-Maximização) até aplicações práticas de agrupamento de dígitos manuscritos e compressão de cores em imagens.

---

## 🔗 Projeto Original

O conteúdo e os códigos contidos nesta pasta são adaptados de:
*   **Notebook Original no Google Colab:** [K-Means Notebook](https://colab.research.google.com/drive/1hrWkkacxdGX7J6dbYC9Fvedrepzcdk33)
*   **Material de Referência:** Jake VanderPlas — [Python Data Science Handbook: K-Means](https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.11-K-Means.ipynb)

Arquivos principais neste diretório:
*   [K_Means.ipynb](file:///c:/Users/your-user/Documents/Formacao-Machine-Learning-Specialist/Algoritmos%20de%20Treinamento%20em%20Machine%20Learning/Treinamento%20N%C3%A3o-Supervisionado%20em%20Machine%20Learning/K_Means.ipynb) — Notebook interativo contendo a demonstração passo a passo do algoritmo.
*   [k_means.py](file:///c:/Users/your-user/Documents/Formacao-Machine-Learning-Specialist/Algoritmos%20de%20Treinamento%20em%20Machine%20Learning/Treinamento%20N%C3%A3o-Supervisionado%20em%20Machine%20Learning/k_means.py) — Script Python adaptado para execução local estruturada.

---

## 💻 Diferenças de Rodar Localmente vs. Google Colab / Original

Ao migrar a execução do ambiente interativo de nuvem (Google Colab) para a máquina local via script Python, foram feitas adaptações importantes para melhor estruturação e reprodutibilidade:

### 1. Exibição de Gráficos e Imagens
*   **No Colab:** As visualizações geradas pela biblioteca `matplotlib` são exibidas imediatamente abaixo das células de código (`%matplotlib inline`).
*   **Localmente:** O script [k_means.py](file:///c:/Users/your-user/Documents/Formacao-Machine-Learning-Specialist/Algoritmos%20de%20Treinamento%20em%20Machine%20Learning/Treinamento%20N%C3%A3o-Supervisionado%20em%20Machine%20Learning/k_means.py) salva todas as figuras automaticamente como imagens `.png` organizadas dentro da pasta de saída [saida_kmeans](file:///c:/Users/your-user/Documents/Formacao-Machine-Learning-Specialist/Algoritmos%20de%20Treinamento%20em%20Machine%20Learning/Treinamento%20N%C3%A3o-Supervisionado%20em%20Machine%20Learning/saida_kmeans). Os gráficos são fechados logo após a gravação com `plt.close()` para evitar consumo excessivo de memória.

### 2. Persistência de Resultados
*   **No Colab:** Os prints de acurácia, dimensões de dados e diagnósticos são exibidos de forma volátil no console do notebook.
*   **Localmente:** Além de printar as informações no terminal, o script cria e escreve os resultados consolidados em um arquivo de log estruturado chamado `resultados.txt` dentro da pasta de saída [saida_kmeans](file:///c:/Users/your-user/Documents/Formacao-Machine-Learning-Specialist/Algoritmos%20de%20Treinamento%20em%20Machine%20Learning/Treinamento%20N%C3%A3o-Supervisionado%20em%20Machine%20Learning/saida_kmeans).

### 3. Gerenciamento de Dependências
*   **No Colab:** O ambiente já vem com quase todas as dependências pré-instaladas por padrão.
*   **Localmente:** É necessário configurar o ambiente local (por exemplo, utilizando o ambiente virtual local `.venv`) e instalar as bibliotecas de ciência de dados necessárias para que o script execute sem erros.

---

## 🚀 Como Executar Localmente

### 1. Ativar o Ambiente Virtual
A partir do diretório raiz do projeto, ative o ambiente virtual configurado:

No Windows (PowerShell):
```powershell
..\..\.venv\Scripts\Activate.ps1
```

### 2. Instalar Dependências (caso não estejam instaladas)
Certifique-se de ter os pacotes instalados:
```bash
pip install numpy matplotlib seaborn scikit-learn scipy pillow
```

### 3. Rodar o Script K-Means
Execute o script diretamente pelo terminal:
```bash
python k_means.py
```

Após a execução bem-sucedida, você poderá verificar a pasta gerada [saida_kmeans](file:///c:/Users/your-user/Documents/Formacao-Machine-Learning-Specialist/Algoritmos%20de%20Treinamento%20em%20Machine%20Learning/Treinamento%20N%C3%A3o-Supervisionado%20em%20Machine%20Learning/saida_kmeans) contendo todas as visualizações (de `01_blobs_originais.png` a `13_china_recolored_comparison.png`) e o arquivo `resultados.txt`.
