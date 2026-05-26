# Algoritmos para Rastreamento de Objetos em Imagens

Este repositório contém as anotações e os conceitos fundamentais sobre o módulo de **Algoritmos para Rastreamento de Objetos em Imagens**.

---

## 📚 Conteúdo das Aulas

### 🎥 Aula 1 - Introdução ao Rastreamento de Objetos
Aborda o conceito básico de rastreamento (*tracking*) de objetos ao longo do tempo.
*   **O que é Rastreamento?** O processo de estimar a trajetória de um objeto de interesse em uma sequência de vídeo (ou frames consecutivos) após a sua localização inicial.
*   **Associação de Dados:** O desafio de ligar as detecções do mesmo objeto em diferentes instantes de tempo, mantendo o mesmo identificador (ID).
*   **Aplicações:** Vigilância inteligente, tráfego urbano, robótica e veículos autônomos.

---

### 🧠 Aula 2 - Redes para Rastreamento de Objetos
Estudo dos limites dos algoritmos de detecção tradicionais e da necessidade de arquiteturas específicas para rastreamento contínuo.
*   **Limitações dos Detectores Tradicionais:** Explicação de por que algoritmos focados puramente em detecção estática nem sempre funcionam corretamente de frame para frame (devido a oclusões temporárias, mudanças abruptas de luz ou desfoque de movimento).
*   **Rastreamento de Obstáculos:** A importância de antecipar e manter o rastreamento em cenários de oclusão parcial ou total.
*   **Tipos de Redes para Tracking em Imagens:**
    *   **YOLO + SORT/DeepSORT:** Integração de detecção baseada em aprendizado profundo com algoritmos de associação espacial e temporal.
    *   **Mask R-CNN:** Rastreamento combinado com segmentação de instâncias para obter máscaras de pixels precisas de cada objeto rastreado.

---

### ⚙️ Aula 3 - Rede YOLO para Rastreamento
Foco em arquiteturas que unem detecção rápida e rastreamento robusto integrando dados no domínio do tempo.
*   **YOLO + SORT:** Detecção eficiente usando YOLO em conjunto com o algoritmo SORT (*Simple Online and Realtime Tracking*), que utiliza Filtro de Kalman e o Algoritmo Húngaro para associar caixas delimitadoras ao longo do tempo.
*   **Detecção + Rastreamento em Função do Tempo:** O uso de informações temporais e histórico de movimento para prever a próxima posição do objeto.
*   **DL + LSTM:** Uso de Aprendizado Profundo (*Deep Learning*) para extração de características visuais combinado com redes Recorrentes (LSTM - *Long Short-Term Memory*) para modelar dependências temporais e trajetórias históricas dos objetos.

---

### 💻 Aula 4 - Redes de Rastreamento na Prática
Implementação prática de algoritmos de rastreamento utilizando modelos pré-treinados e frameworks modernos.
*   **Prática Hands-on:** Desenvolvimento e execução de scripts de rastreamento em tempo real.
*   **Script de Referência:** Implementação prática utilizando o script [yolov4_deepsort.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Visao%20Computacional%20com%20Machine%20Learning/Algoritmos%20para%20Rastreamento%20de%20Objetos%20em%20Imagens/Redes%20de%20rastreamento%20na%20pratica/yolov4_deepsort.py) (que combina o detector YOLOv4 com o algoritmo de rastreamento DeepSORT).
*   **Guia de Configuração Local:** Veja o guia detalhado [YOLOv4_DeepSORT_Local.md](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Visao%20Computacional%20com%20Machine%20Learning/Algoritmos%20para%20Rastreamento%20de%20Objetos%20em%20Imagens/Redes%20de%20rastreamento%20na%20pratica/YOLOv4_DeepSORT_Local.md) descrevendo as alterações realizadas em relação ao notebook do Google Colab para a execução local.
