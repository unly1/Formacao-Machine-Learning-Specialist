# -*- coding: utf-8 -*-
"""
Treinamento de uma Rede Neural do Zero com PyTorch para o dataset MNIST.
"""

from time import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import sys

# Classe para duplicar a saída do terminal em um arquivo de log
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Classe que define a arquitetura da Rede Neural
class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        # Camada de entrada: 28*28 (784) neurônios se ligando a 128
        self.linear1 = nn.Linear(28 * 28, 128)
        # Camada interna 1: 128 neurônios se ligando a 64
        self.linear2 = nn.Linear(128, 64)
        # Camada interna 2: 64 neurônios se ligando a 10 (classes de 0 a 9)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, X):
        X = F.relu(self.linear1(X))  # Função de ativação Relu na primeira camada
        X = F.relu(self.linear2(X))  # Função de ativação Relu na segunda camada
        X = self.linear3(X)          # Camada de saída linear
        return F.log_softmax(X, dim=1)  # Log-Softmax para cálculo da perda (NLLLoss)

# Função para treinar o modelo
def treino(modelo, trainloader, device):
    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5)  # Otimizador SGD
    inicio = time()  # Inicia o timer

    criterio = nn.NLLLoss()  # Função de perda
    EPOCHS = 10
    modelo.train()  # Configura o modelo para modo de treinoke

    for epoch in range(EPOCHS):
        perda_acumulada = 0.0

        for imagens, etiquetas in trainloader:
            # Redimensiona as imagens de [batch_size, 1, 28, 28] para [batch_size, 784]
            imagens = imagens.view(imagens.shape[0], -1)
            
            otimizador.zero_grad()  # Zera os gradientes acumulados
            
            output = modelo(imagens.to(device))  # Passagem forward
            perda_instantanea = criterio(output, etiquetas.to(device))  # Calcula a perda
            
            perda_instantanea.backward()  # Backpropagation (gradientes)
            otimizador.step()  # Atualiza os pesos e bias
            
            perda_acumulada += perda_instantanea.item()

        print(f"Época {epoch + 1:02d}/{EPOCHS:02d} - Perda resultante: {perda_acumulada / len(trainloader):.6f}")

    tempo_total = (time() - inicio) / 60
    print(f"Tempo de treino (em minutos) = {tempo_total:.2f}")

# Função para validação/avaliação do modelo
def validacao(modelo, valloader, device):
    modelo.eval()  # Configura o modelo para modo de avaliação
    conta_corretas, conta_todas = 0, 0

    for imagens, etiquetas in valloader:
        for i in range(len(etiquetas)):
            # Redimensiona a imagem para entrada da rede
            img = imagens[i].view(1, 784)
            
            # Desativa o autograd para acelerar a validação e economizar memória
            with torch.no_grad():
                logps = modelo(img.to(device))  # Output em escala logarítmica
            
            ps = torch.exp(logps)  # Converte o output para escala de probabilidade
            probab = list(ps.cpu().numpy()[0])
            etiqueta_pred = probab.index(max(probab))  # Pega a classe com maior probabilidade
            etiqueta_certa = etiquetas.numpy()[i]
            
            if etiqueta_certa == etiqueta_pred:
                conta_corretas += 1
            conta_todas += 1

    precisao = (conta_corretas * 100) / conta_todas
    print(f"Total de imagens testadas = {conta_todas}")
    print(f"Precisão do modelo = {precisao:.2f}%")

if __name__ == "__main__":
    # Configura o Logger para salvar o resultado do terminal em 'resultado.txt' no mesmo diretório do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "resultado.txt")
    sys.stdout = Logger(log_path)

    # Configuração de transformações para normalização e conversão para tensor
    transform = transforms.ToTensor()

    # Carrega os dados de treino e validação
    print("Carregando o dataset MNIST...")
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    valset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    # Visualização rápida de uma imagem de treino
    dataiter = iter(trainloader)
    imagens, etiquetas = next(dataiter)
    
    print(f"Dimensões do tensor de imagens: {imagens[0].shape}")
    print(f"Dimensões do tensor de etiquetas: {etiquetas[0].shape}")
    
    # Exibe a imagem de amostra (descomente plt.show() se quiser visualizar localmente)
    plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')
    plt.title(f"Exemplo de treino - Rótulo: {etiquetas[0].item()}")
    # plt.show()

    # Inicialização do modelo e detecção de GPU (CUDA)
    modelo = Modelo()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    print(f"Modelo enviado para o dispositivo: {device}")

    # Executa o treino e a validação
    print("\n--- Iniciando Treinamento ---")
    treino(modelo, trainloader, device)

    print("\n--- Iniciando Validação ---")
    validacao(modelo, valloader, device)

    # Restaura o stdout padrão e fecha o arquivo de log
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal