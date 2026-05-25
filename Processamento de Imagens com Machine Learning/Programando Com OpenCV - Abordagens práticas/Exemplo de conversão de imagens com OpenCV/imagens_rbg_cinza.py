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
            print(
                f"Imagem baixada e salva com sucesso em: '{caminho_completo}'")
        except Exception as e:
            print(f"Erro ao baixar a imagem: {e}")
            return None
    else:
        print(f"Imagem já existente em: '{caminho_completo}'")

    return caminho_completo


def main():
    # URL da imagem da Lenna
    url_lenna = "https://pns2019.github.io/images/Lenna.png"
    pasta_imagens = "imagens"
    nome_arquivo = "Lenna.png"

    # Baixar a imagem
    caminho_imagem = baixar_imagem(url_lenna, pasta_imagens, nome_arquivo)

    if caminho_imagem and os.path.exists(caminho_imagem):
        # Carregar a imagem com OpenCV (formato BGR padrão)
        img_bgr = cv2.imread(caminho_imagem)

        if img_bgr is not None:
            # 1. Converter de BGR (OpenCV) para RGB (Matplotlib) para exibição correta
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 2. Converter a imagem de BGR para Escala de Cinza
            img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Criar pasta para salvar os resultados
            pasta_resultados = "resultados"
            os.makedirs(pasta_resultados, exist_ok=True)

            # Salvar a imagem em escala de cinza no disco
            caminho_cinza = os.path.join(pasta_resultados, "lenna_cinza.png")
            cv2.imwrite(caminho_cinza, img_cinza)
            print(
                f"Imagem em escala de cinza salva com sucesso em: '{caminho_cinza}'")

            # Configurar exibição comparativa usando Matplotlib
            plt.figure(figsize=(10, 5))

            # Exibir Imagem Colorida (RGB)
            plt.subplot(1, 2, 1)
            plt.imshow(img_rgb)
            plt.title("Imagem Colorida (RGB)")
            plt.axis("off")

            # Exibir Imagem em Escala de Cinza
            plt.subplot(1, 2, 2)
            plt.imshow(img_cinza, cmap="gray")
            plt.title("Imagem em Escala de Cinza")
            plt.axis("off")

            plt.tight_layout()

            # Salvar o gráfico comparativo
            caminho_comparativo = os.path.join(
                pasta_resultados, "comparativo_rgb_cinza.png")
            plt.savefig(caminho_comparativo, dpi=300)
            print(f"Gráfico comparativo salvo em: '{caminho_comparativo}'")

            # Exibir a janela gráfica se possível
            try:
                plt.show()
            except Exception as e:
                print(
                    f"Não foi possível abrir a janela do Matplotlib (provavelmente por falta de ambiente gráfico): {e}")
        else:
            print("Erro ao decodificar a imagem carregada com OpenCV.")
    else:
        print("Erro: Não foi possível obter ou carregar a imagem da Lenna.")


if __name__ == "__main__":
    main()
