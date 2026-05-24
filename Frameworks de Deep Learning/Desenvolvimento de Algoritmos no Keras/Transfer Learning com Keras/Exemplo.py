import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# Obter o caminho absoluto para o arquivo de imagem na pasta 'imagem'
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'imagem', 'gato.png')

# 1. Carregar a base do modelo pré-treinado (ResNet50) sem a camada de classificação final (include_top=False)
print("Carregando o modelo base ResNet50 (congelado)...")
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))

# 2. Congelar os pesos da base pré-treinada para que não sejam atualizados durante o treinamento preliminar
base_model.trainable = False

# 3. Adicionar novas camadas personalizadas no topo (Transfer Learning)
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    # Exemplo com 2 classes personalizadas
    layers.Dense(2, activation='softmax')
])

# 4. Compilar o modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nResumo do modelo de Transfer Learning:")
model.summary()

# 5. Carregar e preparar a imagem para teste
if os.path.exists(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extrair os recursos (features) do modelo base
    features = base_model.predict(x)
    print("\nShape dos recursos extraídos pelo modelo base congelado:", features.shape)

    # Executar a predição no novo modelo personalizado
    preds = model.predict(x)
    print("Predição (probabilidade para cada uma das 2 novas classes):", preds)

    # Salvar os resultados em um arquivo .txt
    output_path = os.path.join(script_dir, 'resultado_transfer_learning.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Resultado do Script de Transfer Learning:\n")
        f.write(f"- Shape das features extraídas: {features.shape}\n")
        f.write(f"- Classes de saída no novo classificador: 2\n")
        f.write(f"- Modelo base (ResNet50) congelado: Sim\n")
        f.write(f"- Predição para a imagem teste: {preds[0].tolist()}\n")
    print(f"Resultados salvos com sucesso em: {output_path}")
else:
    print(f"Erro: Imagem não encontrada em {img_path}")
