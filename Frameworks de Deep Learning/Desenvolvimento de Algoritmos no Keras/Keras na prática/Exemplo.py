import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Obter o caminho absoluto para o arquivo de imagem na pasta 'imagem'
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'imagem', 'gato.png')

model = ResNet50(weights='imagenet')

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]
print('Predicted:', decoded_preds)

# Salvar os resultados em um arquivo .txt no mesmo diretório do script
output_path = os.path.join(script_dir, 'resultado.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("Resultado das Predicoes (ResNet50):\n")
    for i, (imagenet_id, label, prob) in enumerate(decoded_preds):
        f.write(f"{i+1}. {label} ({imagenet_id}): {prob*100:.2f}%\n")

print(f"Resultados salvos com sucesso em: {output_path}")

plt.imshow(img)
plt.axis('off')
plt.show()
