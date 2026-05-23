# Guia de Execução Local - Validação Cruzada (Cross-Validation)

Este diretório contém o script `cross_validation.py`, que demonstra a aplicação de técnicas de validação cruzada (KFold, cross_val_score e LOOCV) usando a biblioteca `scikit-learn` em conjuntos de dados clássicos.

> [!NOTE]
> Este projeto é uma adaptação do material original desenvolvido por Chanseok Kang, disponível no notebook [Google Colab - Cross-Validation](https://colab.research.google.com/github/goodboychan/chans_jupyter/blob/main/_notebooks/2020-07-14-01-Cross-Validation.ipynb#scrollTo=qe3sEnETPlE5).

---

## 🛠️ O que foi feito

Para atender aos requisitos de exportação de resultados, foram realizadas as seguintes modificações em [cross_validation.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/M%C3%A9todos%20de%20Valida%C3%A7%C3%A3o%20de%20Treinamento/cross_validation.py):

1. **Redirecionamento Dinâmico de Saída (Tee):**
   * Foi implementada uma classe customizada chamada `Tee` para duplicar a saída padrão (`sys.stdout`).
   * Toda informação impressa no console com `print()` é automaticamente gravada em tempo de execução em uma pasta dedicada de resultados, gerando o arquivo [resultados_cross_validation.txt](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/M%C3%A9todos%20de%20Valida%C3%A7%C3%A3o%20de%20Treinamento/resultados/resultados_cross_validation.txt).

2. **Gerenciamento de Recursos robusto:**
   * Utilizou-se o módulo `atexit` para garantir que o arquivo de texto seja fechado de forma limpa ao fim da execução do script.
   * A classe `Tee` foi ajustada para tratar o encerramento do interpretador sem lançar exceções de E/S (`ValueError: I/O operation on closed file`).

---

## 🚀 Como Rodar Localmente

Siga o passo a passo abaixo para executar o script em sua máquina local usando o ambiente virtual configurado:

### 1. Abrir o Terminal no Diretório Raiz
Abra o seu terminal (preferencialmente PowerShell no Windows) no diretório raiz do projeto:
```bash
cd "c:\Users\moliv\Documents\Formacao-Machine-Learning-Specialist"
```

### 2. Ativar o Ambiente Virtual (`.venv`)
Execute o comando correspondente ao seu terminal para ativar o ambiente virtual:

* **No PowerShell:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **No Prompt de Comando (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **No Git Bash / Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Instalar Dependências (se necessário)
Caso ainda não as tenha instaladas no ambiente virtual, certifique-se de instalar as dependências requeridas pelo script:
```bash
pip install pandas numpy matplotlib scikit-learn
```

### 4. Executar o Script
Rode o script utilizando o interpretador do ambiente virtual:
```bash
python "Teoria do Aprendizado Estatístico/Métodos de Validação de Treinamento/cross_validation.py"
```

---

## 📄 Resultados Gerados

Após a execução bem-sucedida, você verá as saídas no console e um novo arquivo de texto será gerado em uma pasta dedicada do script:
* **Pasta de saída:** [resultados/](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/M%C3%A9todos%20de%20Valida%C3%A7%C3%A3o%20de%20Treinamento/resultados/)
* **Arquivo:** [resultados_cross_validation.txt](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/M%C3%A9todos%20de%20Valida%C3%A7%C3%A3o%20de%20Treinamento/resultados/resultados_cross_validation.txt)
* **Conteúdo:** Acurácia de divisão, proporção de classes, erros médios do cross-validation e métricas do LOOCV (Leave-One-Out Cross-Validation).
