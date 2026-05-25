# Filtros de Eliminação de Ruídos em Imagens Digitais

Este projeto demonstra na prática como simular e eliminar ruídos comuns em imagens digitais utilizando a biblioteca OpenCV em Python.

## 📌 Visão Geral do Script

O script [elimina_ruido.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Processamento%20de%20Imagens%20com%20Machine%20Learning/Filtros%20de%20Elimina%C3%A7%C3%A3o%20de%20ru%C3%ADdos%20em%20imagens%20digitais/Tratamento%20de%20ru%C3%ADdos%20em%20imagens%20digitais/elimina_ruido.py) realiza o download automático da imagem de teste clássica **Lena**, aplica artificialmente dois tipos frequentes de ruídos, e depois utiliza filtros específicos para tentar remover esses ruídos, gerando gráficos comparativos de desempenho.

---

## 🌪️ Tipos de Ruídos Simulados

1. **Ruído Gaussiano (Gaussian Noise):**
   - **O que é:** Ruído eletrônico aleatório que segue uma distribuição normal (curva em formato de sino). Afeta todos os pixels da imagem em graus variados.
   - **Causa comum:** Sensores de câmeras digitais operando sob pouca iluminação ou alta temperatura.

2. **Ruído Sal e Pimenta (Salt and Pepper Noise):**
   - **O que é:** Ruído impulsivo composto por pixels pretos (pimenta) e brancos (sal) distribuídos aleatoriamente pela imagem.
   - **Causa comum:** Falhas na transmissão digital de dados, problemas de conversão analógico-digital ou defeitos de hardware no sensor de imagem.

---

## 🛡️ Filtros de Eliminação de Ruídos

O OpenCV disponibiliza vários filtros clássicos, cada um com comportamentos distintos dependendo do ruído:

* **Filtro de Média (Averaging Blur - `cv2.blur`):**
  Suaviza a imagem substituindo cada pixel pela média dos pixels vizinhos em um kernel retangular. É simples, mas desfoca bordas.
  
* **Filtro Gaussiano (Gaussian Blur - `cv2.GaussianBlur`):**
  Utiliza um peso ponderado (distribuição gaussiana) baseado na distância ao pixel central. Oferece uma suavização mais natural que o filtro de média, mas ainda atenua bordas importantes.

* **Filtro de Mediana (Median Blur - `cv2.medianBlur`):**
  Substitui o pixel central pelo valor mediano de sua vizinhança. **Extremamente eficaz contra Ruído Sal e Pimenta**, pois os pontos extremos de ruído (0 ou 255) são descartados na ordenação mediana.

* **Filtro Bilateral (`cv2.bilateralFilter`):**
  Considera tanto a proximidade espacial quanto a similaridade de intensidade dos pixels. **Preserva as bordas com precisão** ao mesmo tempo em que suaviza áreas homogêneas, sendo excelente contra ruído gaussiano.

---

## 🚀 Como Executar o Script

### Pré-requisitos
Certifique-se de que possui as bibliotecas necessárias instaladas em seu ambiente virtual:
```bash
pip install opencv-python numpy matplotlib
```

### Execução
Navegue até a pasta do projeto e execute:
```bash
python elimina_ruido.py
```

---

## 📁 Resultados Gerados

Após a execução, o script criará duas pastas no mesmo diretório do arquivo:
1. `imagens/`: Contém a imagem original baixada (`lena.jpg`).
2. `resultados/`: Contém as imagens individuais com ruído, cada uma das imagens após a aplicação dos filtros e os gráficos comparativos:
   * `comparativo_ruido_gaussiano.png`: Mostra a imagem original, ruidosa e o resultado dos 4 filtros.
   * `comparativo_ruido_sal_pimenta.png`: Mostra o contraste evidente entre o filtro Gaussiano (inadequado) e o filtro de Mediana (ideal) para corrigir ruído do tipo sal e pimenta.
