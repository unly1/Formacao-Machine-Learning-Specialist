# 🎯 Treinamento Supervisionado para Machine Learning — Classificação com Scikit-Learn

Este diretório contém a implementação prática estruturada para o treinamento, validação e persistência de um modelo de **Aprendizado Supervisionado** (Classificação) utilizando a biblioteca **Scikit-Learn** em Python.

---

## 📁 Estrutura do Diretório

O diretório é composto pelos seguintes elementos principais:
*   **[classificacao_sklearn.py](classificacao_sklearn.py)**: Script principal em Python contendo o pipeline completo de Machine Learning (carga, divisão de dados, normalização, treinamento, avaliação e persistência).
*   **[resultado/](resultado)**: Pasta criada automaticamente ao executar o script para armazenar os artefatos gerados pelo treinamento.
    *   `modelo_classificacao.pkl`: O classificador RandomForest treinado e serializado.
    *   `scaler.pkl`: O objeto padronizador (`StandardScaler`) treinado e serializado.
    *   `resultado_classificacao.txt`: Relatório completo em formato texto contendo os prints da execução e métricas de desempenho.

---

## ⚙️ Fluxo do Treinamento Básico

O script [classificacao_sklearn.py](classificacao_sklearn.py) realiza um pipeline padrão de modelagem supervisionada:

### 1. Carregamento dos Dados
Utiliza a base de dados clássica **Iris** (`load_iris()`), que contém 150 amostras e 3 classes de flores. É possível adaptar o código facilmente para ler arquivos externos (como `.csv`) usando `pandas.read_csv()`.

### 2. Divisão Treino e Teste
Divide os dados em conjuntos de:
*   **Treino (80%)**: Utilizado para o modelo aprender os padrões.
*   **Teste (20%)**: Reservado para avaliar o desempenho final em dados não vistos.
*   *Nota:* O uso de `stratify=y` garante que a proporção das classes originais seja mantida em ambas as divisões, e `random_state=42` garante a reprodutibilidade.

### 3. Pré-processamento (Normalização)
Instancia um `StandardScaler` para reescalar as features para média 0 e desvio padrão 1.
*   O ajuste (`.fit_transform()`) é feito **apenas** sobre os dados de treino para evitar vazamento de dados (*data leakage*).
*   Os dados de teste são apenas transformados (`.transform()`) com base no scaler já ajustado.

### 4. Instanciação e Treinamento do Modelo
Instancia o classificador **Random Forest** (`RandomForestClassifier`) com 100 árvores de decisão. O treinamento é realizado chamando o método `.fit(X_train, y_train)`.

### 5. Avaliação do Modelo
Gera previsões para o conjunto de teste e exibe três métricas principais no console:
*   **Acurácia**: Taxa geral de acertos.
*   **Relatório de Classificação**: Detalhamento de métricas como Precision, Recall e F1-score por classe.
*   **Matriz de Confusão**: Tabela comparativa entre valores reais e preditos para mapear erros e acertos.

---

## 💾 Salvamento e Persistência (Salvamento Básico)

Um ponto fundamental no ciclo de vida de Machine Learning é salvar o modelo para que possa ser utilizado posteriormente em produção sem a necessidade de re-treiná-lo.

No script, isso é feito utilizando a biblioteca `joblib` para serializar e salvar os arquivos no disco:

```python
import joblib

# Salvando o modelo e o scaler
joblib.dump(modelo, "resultado/modelo_classificacao.pkl")
joblib.dump(scaler, "resultado/scaler.pkl")
```

### Exemplo de Inferência (Carregando o Modelo Salvo)
Para usar o modelo salvo em novos dados:

```python
import joblib
import numpy as np

# 1. Carrega os artefatos salvos
modelo_carregado = joblib.load("resultado/modelo_classificacao.pkl")
scaler_carregado = joblib.load("resultado/scaler.pkl")

# 2. Define uma nova amostra (com as features originais sem escala)
nova_amostra = np.array([[5.1, 3.5, 1.4, 0.2]])

# 3. Aplica o mesmo scaler do treinamento
nova_amostra_scaled = scaler_carregado.transform(nova_amostra)

# 4. Realiza a predição da classe
predicao = modelo_carregado.predict(nova_amostra_scaled)
print(f"Classe predita (código): {predicao[0]}")
```

---

## 🚀 Como Executar Localmente

### 1. Ativar o Ambiente Virtual
A partir desta pasta, ative o ambiente virtual configurado no projeto:

No Windows (PowerShell):
```powershell
..\..\.venv\Scripts\Activate.ps1
```

### 2. Instalar Bibliotecas
Certifique-se de que os pacotes necessários estão instalados:
```bash
pip install numpy scikit-learn joblib
```

### 3. Executar o Script
Rode o script pelo console:
```bash
python classificacao_sklearn.py
```

Após a finalização, verifique os artefatos salvos dentro do diretório [resultado/](resultado).
