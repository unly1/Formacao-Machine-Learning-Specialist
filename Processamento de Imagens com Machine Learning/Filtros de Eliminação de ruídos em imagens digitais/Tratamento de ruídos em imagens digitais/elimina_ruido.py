import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request


def baixar_imagem(url, pasta_destino, nome_arquivo):
    """
    Cria a pasta de destino caso não exista e realiza o download da imagem a partir da URL.
    """
    os.makedirs(pasta_destino, exist_ok=True)
    caminho_completo = os.path.join(pasta_destino, nome_arquivo)

    if not os.path.exists(caminho_completo):
        print(f"Baixando imagem de: {url}...")
        try:
            # Usar um cabeçalho User-Agent para evitar problemas de requisição bloqueada
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                with open(caminho_completo, 'wb') as f:
                    f.write(response.read())
            print(f"Imagem baixada e salva com sucesso em: '{caminho_completo}'")
        except Exception as e:
            print(f"Erro ao baixar a imagem: {e}")
            return None
    else:
        print(f"Imagem já existente em: '{caminho_completo}'")

    return caminho_completo


def adicionar_ruido_gaussiano(imagem, media=0, desvio_padrao=25):
    """
    Adiciona ruído Gaussiano à imagem.
    Ideal para simular ruídos de sensores de câmera sob baixa luminosidade.
    """
    ruido = np.random.normal(media, desvio_padrao, imagem.shape)
    imagem_ruidosa = imagem.astype(np.float64) + ruido
    return np.clip(imagem_ruidosa, 0, 255).astype(np.uint8)


def adicionar_ruido_sal_pimenta(imagem, proporcao=0.04):
    """
    Adiciona ruído Sal e Pimenta (Salt and Pepper) à imagem.
    Simula falhas de transmissão de bits ou pixels defeituosos no sensor (CCD).
    """
    imagem_ruidosa = imagem.copy()
    
    # Sal (pixels brancos - 255)
    num_sal = np.ceil(proporcao * imagem.size * 0.5)
    coords_sal_y = np.random.randint(0, imagem.shape[0] - 1, int(num_sal))
    coords_sal_x = np.random.randint(0, imagem.shape[1] - 1, int(num_sal))
    if len(imagem.shape) == 3:
        imagem_ruidosa[coords_sal_y, coords_sal_x, :] = 255
    else:
        imagem_ruidosa[coords_sal_y, coords_sal_x] = 255

    # Pimenta (pixels pretos - 0)
    num_pimenta = np.ceil(proporcao * imagem.size * 0.5)
    coords_pimenta_y = np.random.randint(0, imagem.shape[0] - 1, int(num_pimenta))
    coords_pimenta_x = np.random.randint(0, imagem.shape[1] - 1, int(num_pimenta))
    if len(imagem.shape) == 3:
        imagem_ruidosa[coords_pimenta_y, coords_pimenta_x, :] = 0
    else:
        imagem_ruidosa[coords_pimenta_y, coords_pimenta_x] = 0

    return imagem_ruidosa


def salvar_imagem(caminho, imagem):
    """
    Salva a imagem no disco de forma robusta, suportando caracteres Unicode no caminho do Windows.
    """
    extensão = os.path.splitext(caminho)[1]
    sucesso, buffer = cv2.imencode(extensão, imagem)
    if sucesso:
        buffer.tofile(caminho)
        return True
    return False


def main():
    # Obter o diretório atual do script
    diretorio_script = os.path.dirname(os.path.abspath(__file__))

    # URL de imagem de exemplo do OpenCV (Lena)
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    pasta_imagens = os.path.join(diretorio_script, "imagens")
    nome_arquivo = "lena.jpg"

    # Baixar a imagem original
    caminho_imagem = baixar_imagem(url, pasta_imagens, nome_arquivo)

    if not caminho_imagem or not os.path.exists(caminho_imagem):
        print("Erro: Não foi possível obter a imagem para processamento.")
        return

    # Carregar imagem original (suporta caracteres Unicode no caminho)
    img_bgr = cv2.imdecode(np.fromfile(caminho_imagem, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img_bgr is None:
        print("Erro ao decodificar a imagem com OpenCV.")
        return

    # Converter para RGB para exibição correta com Matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Criar pasta para salvar os resultados
    pasta_resultados = os.path.join(diretorio_script, "resultados")
    os.makedirs(pasta_resultados, exist_ok=True)

    # ----------------------------------------------------
    # Parte 1: Simulação e Tratamento de Ruído Gaussiano
    # ----------------------------------------------------
    print("\n--- Processando Ruído Gaussiano ---")
    img_gaussiana = adicionar_ruido_gaussiano(img_rgb, desvio_padrao=25)
    salvar_imagem(os.path.join(pasta_resultados, "1_ruido_gaussiano.png"), cv2.cvtColor(img_gaussiana, cv2.COLOR_RGB2BGR))

    # Filtro de Média (Averaging)
    # Suaviza a imagem inteira uniformemente. Kernel 5x5.
    filtro_media_gauss = cv2.blur(img_gaussiana, (5, 5))
    salvar_imagem(os.path.join(pasta_resultados, "2_gauss_filtro_media.png"), cv2.cvtColor(filtro_media_gauss, cv2.COLOR_RGB2BGR))

    # Filtro Gaussiano
    # Pondera pixels próximos mais fortemente que os distantes. Kernel 5x5, sigma=1.5.
    filtro_gaussiano = cv2.GaussianBlur(img_gaussiana, (5, 5), 1.5)
    salvar_imagem(os.path.join(pasta_resultados, "3_gauss_filtro_gaussiano.png"), cv2.cvtColor(filtro_gaussiano, cv2.COLOR_RGB2BGR))

    # Filtro de Mediana
    # Substitui pelo valor mediano da vizinhança. Kernel tamanho 5.
    filtro_mediana_gauss = cv2.medianBlur(img_gaussiana, 5)
    salvar_imagem(os.path.join(pasta_resultados, "4_gauss_filtro_mediana.png"), cv2.cvtColor(filtro_mediana_gauss, cv2.COLOR_RGB2BGR))

    # Filtro Bilateral
    # Suaviza o ruído preservando bordas (compara intensidade de pixels além da distância espacial).
    filtro_bilateral_gauss = cv2.bilateralFilter(img_gaussiana, d=9, sigmaColor=75, sigmaSpace=75)
    salvar_imagem(os.path.join(pasta_resultados, "5_gauss_filtro_bilateral.png"), cv2.cvtColor(filtro_bilateral_gauss, cv2.COLOR_RGB2BGR))

    # Plot Comparativo do Ruído Gaussiano
    plt.figure(figsize=(15, 10))
    imagens_gauss = [img_rgb, img_gaussiana, filtro_media_gauss, filtro_gaussiano, filtro_mediana_gauss, filtro_bilateral_gauss]
    titulos_gauss = [
        "Imagem Original", "Ruído Gaussiano", 
        "Filtro Média (Averaging)", "Filtro Gaussiano", 
        "Filtro Mediana", "Filtro Bilateral (Preserva Bordas)"
    ]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(imagens_gauss[i])
        plt.title(titulos_gauss[i])
        plt.axis("off")

    plt.tight_layout()
    caminho_plot_gauss = os.path.join(pasta_resultados, "comparativo_ruido_gaussiano.png")
    plt.savefig(caminho_plot_gauss, dpi=300)
    print(f"Gráfico comparativo de ruído Gaussiano salvo em: '{caminho_plot_gauss}'")

    # ----------------------------------------------------
    # Parte 2: Simulação e Tratamento de Ruído Sal e Pimenta
    # ----------------------------------------------------
    print("\n--- Processando Ruído Sal e Pimenta ---")
    img_sal_pimenta = adicionar_ruido_sal_pimenta(img_rgb, proporcao=0.05)
    salvar_imagem(os.path.join(pasta_resultados, "6_ruido_sal_pimenta.png"), cv2.cvtColor(img_sal_pimenta, cv2.COLOR_RGB2BGR))

    # Filtro Gaussiano aplicado ao Sal e Pimenta (Geralmente ruim)
    filtro_gaussiano_sp = cv2.GaussianBlur(img_sal_pimenta, (5, 5), 1.5)
    salvar_imagem(os.path.join(pasta_resultados, "7_sp_filtro_gaussiano.png"), cv2.cvtColor(filtro_gaussiano_sp, cv2.COLOR_RGB2BGR))

    # Filtro de Mediana aplicado ao Sal e Pimenta (Ideal/Excelente)
    filtro_mediana_sp = cv2.medianBlur(img_sal_pimenta, 5)
    salvar_imagem(os.path.join(pasta_resultados, "8_sp_filtro_mediana.png"), cv2.cvtColor(filtro_mediana_sp, cv2.COLOR_RGB2BGR))

    # Plot Comparativo do Ruído Sal e Pimenta
    plt.figure(figsize=(15, 5))
    imagens_sp = [img_rgb, img_sal_pimenta, filtro_gaussiano_sp, filtro_mediana_sp]
    titulos_sp = [
        "Imagem Original", "Ruído Sal e Pimenta (5%)", 
        "Filtro Gaussiano (Insuficiente)", "Filtro Mediana (Ideal)"
    ]

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(imagens_sp[i])
        plt.title(titulos_sp[i])
        plt.axis("off")

    plt.tight_layout()
    caminho_plot_sp = os.path.join(pasta_resultados, "comparativo_ruido_sal_pimenta.png")
    plt.savefig(caminho_plot_sp, dpi=300)
    print(f"Gráfico comparativo de ruído Sal e Pimenta salvo em: '{caminho_plot_sp}'")

    print("\nProcessamento concluído com sucesso!")
    print(f"Todos os resultados foram salvos na pasta: '{pasta_resultados}'")

    # Exibição interativa das janelas (se houver suporte de tela)
    try:
        plt.show()
    except Exception as e:
        print(f"\nNota: Não foi possível exibir o gráfico interativo (ambiente gráfico ausente): {e}")


if __name__ == "__main__":
    main()
