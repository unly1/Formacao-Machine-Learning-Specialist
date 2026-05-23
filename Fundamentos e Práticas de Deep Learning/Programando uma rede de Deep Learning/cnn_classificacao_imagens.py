import os
import datetime
import matplotlib
matplotlib.use('Agg')  # salva o gráfico sem precisar de janela gráfica
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import mnist
from keras.utils import to_categorical

# ─────────────────────────────────────────────
# Pasta de resultados
# ─────────────────────────────────────────────
PASTA_RESULTADOS = os.path.join(os.path.dirname(__file__), 'resultados')
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

# ─────────────────────────────────────────────
# Carrega o dataset MNIST
# ─────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape para formato (amostras, altura, largura, canais)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# Dado que o range de valores possível pra um pixel vai de 0-255,
# escalonamos os valores entre 0-1.
# Esse processo torna nosso modelo menos variante a pequenas alterações.
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# One-hot encoding dos rótulos
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test,  10)

# ─────────────────────────────────────────────
# Arquitetura da CNN
# ─────────────────────────────────────────────
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu',
          padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Usada na camada de saída do classificador, onde realmente estamos tentando
# gerar as probabilidades para definir a classe de cada entrada.

optimizer = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

# Reduz o learning rate se não houver melhoras em determinado número de épocas.
# Útil para encontrar o mínimo global.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

batch_size = 32
epochs = 10

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[learning_rate_reduction])

# ─────────────────────────────────────────────
# Avaliação no conjunto de teste
# ─────────────────────────────────────────────
loss_teste, acc_teste = model.evaluate(x_test, y_test, verbose=0)
print(f'\nAcurácia no conjunto de teste: {acc_teste:.4f}')

# ─────────────────────────────────────────────
# Histórico de métricas
# ─────────────────────────────────────────────
history_dict = history.history
acc      = history_dict['accuracy']
val_acc  = history_dict['val_accuracy']
loss     = history_dict['loss']
val_loss = history_dict['val_loss']
range_epochs = range(1, len(acc) + 1)

# ─────────────────────────────────────────────
# Gráfico de acurácia → salvo na pasta resultados/
# ─────────────────────────────────────────────
plt.style.use('default')
plt.figure(figsize=(8, 5))
plt.plot(range_epochs, val_acc, linewidth=2.0, label='Acurácia — Validação')
plt.plot(range_epochs, acc,     linewidth=2.0, label='Acurácia — Treino', color='r')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.title('Acurácia por Época')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PASTA_RESULTADOS, 'grafico_acuracia.png'))
plt.close()
print('Gráfico salvo em resultados/grafico_acuracia.png')

# ─────────────────────────────────────────────
# Arquivo .txt com resumo dos resultados
# ─────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
caminho_txt = os.path.join(PASTA_RESULTADOS, 'resultados.txt')

with open(caminho_txt, 'w', encoding='utf-8') as f:
    f.write('=' * 50 + '\n')
    f.write('  RESULTADOS — CNN Classificação de Imagens\n')
    f.write('=' * 50 + '\n')
    f.write(f'Data/Hora: {timestamp}\n\n')

    f.write(f'Épocas treinadas : {epochs}\n')
    f.write(f'Batch size       : {batch_size}\n\n')

    f.write('--- Histórico por época ---\n')
    f.write(f'{"Época":<8} {"Acc Treino":>12} {"Acc Val":>10} {"Loss Treino":>13} {"Loss Val":>10}\n')
    f.write('-' * 55 + '\n')
    for i in range(len(acc)):
        f.write(f'{i+1:<8} {acc[i]:>12.4f} {val_acc[i]:>10.4f} '
                f'{loss[i]:>13.4f} {val_loss[i]:>10.4f}\n')

    f.write('\n--- Avaliação final no conjunto de TESTE ---\n')
    f.write(f'Loss     : {loss_teste:.4f}\n')
    f.write(f'Acurácia : {acc_teste:.4f} ({acc_teste*100:.2f}%)\n')
    f.write('=' * 50 + '\n')

print(f'Resultados salvos em: {caminho_txt}')
