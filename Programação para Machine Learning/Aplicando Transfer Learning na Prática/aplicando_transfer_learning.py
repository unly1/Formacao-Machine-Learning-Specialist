# -*- coding: utf-8 -*-

import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from shutil import copyfile

# If the URL doesn't work, visit https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# And right click on the 'Download Manually' link to get a new URL to the dataset

# Note: This is a very large dataset and will take time to download

import ssl
import urllib.request

# Define a cross-platform temp directory under the script's folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(SCRIPT_DIR, 'tmp')
os.makedirs(base_dir, exist_ok=True)

url = "https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip"
local_zip = os.path.join(base_dir, 'cats-and-dogs.zip')

# Download the file if it does not already exist to save time
if not os.path.exists(local_zip):
    print(f"Downloading dataset from {url}...")
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore
    urllib.request.urlretrieve(url, local_zip)
    print("Download finished!")
else:
    print("Dataset already downloaded.")

zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(base_dir)
zip_ref.close()

# Define source and destination directories
CAT_SOURCE_DIR = os.path.join(base_dir, "PetImages", "Cat")
DOG_SOURCE_DIR = os.path.join(base_dir, "PetImages", "Dog")

print("Number of cat images:", len(os.listdir(CAT_SOURCE_DIR)))
print("Number of dog images:", len(os.listdir(DOG_SOURCE_DIR)))

# Expected Output
# 12501
# 12501

cats_v_dogs_dir = os.path.join(base_dir, 'cats-v-dogs')
TRAINING_CATS_DIR = os.path.join(cats_v_dogs_dir, "training", "cats")
TESTING_CATS_DIR = os.path.join(cats_v_dogs_dir, "testing", "cats")
TRAINING_DOGS_DIR = os.path.join(cats_v_dogs_dir, "training", "dogs")
TESTING_DOGS_DIR = os.path.join(cats_v_dogs_dir, "testing", "dogs")

os.makedirs(TRAINING_CATS_DIR, exist_ok=True)
os.makedirs(TESTING_CATS_DIR, exist_ok=True)
os.makedirs(TRAINING_DOGS_DIR, exist_ok=True)
os.makedirs(TESTING_DOGS_DIR, exist_ok=True)

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
  files = []
  for filename in os.listdir(SOURCE):
    file = os.path.join(SOURCE, filename)
    if os.path.getsize(file) > 0:
      files.append(filename)
    else:
      print(filename + " is zero length, so ignoring.")

  training_length = int(len(files) * SPLIT_SIZE)
  testing_length = len(files) - training_length
  shuffled_set = random.sample(files, len(files))
  training_set = shuffled_set[0:training_length]
  testing_set = shuffled_set[-testing_length:]

  for filename in training_set:
    this_file = os.path.join(SOURCE, filename)
    destination = os.path.join(TRAINING, filename)
    copyfile(this_file, destination)

  for filename in testing_set:
    this_file = os.path.join(SOURCE, filename)
    destination = os.path.join(TESTING, filename)
    copyfile(this_file, destination)

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print("Training cats:", len(os.listdir(TRAINING_CATS_DIR)))
print("Training dogs:", len(os.listdir(TRAINING_DOGS_DIR)))
print("Testing cats:", len(os.listdir(TESTING_CATS_DIR)))
print("Testing dogs:", len(os.listdir(TESTING_DOGS_DIR)))

# Expected output
# 11250
# 11250
# 1250
# 1250

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = os.path.join(cats_v_dogs_dir, "training")
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=250,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
VALIDATION_DIR = os.path.join(cats_v_dogs_dir, "testing")
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=250,
                                                              class_mode='binary',
                                                              target_size=(150, 150))

# Expected Output:
# Found 23629 images belonging to 2 classes.
# Found 3628 images belonging to 2 classes.

# Note that this may take some time.
history = model.fit(train_generator, epochs=15, steps_per_epoch=90, validation_data=validation_generator, validation_steps=6)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
# ----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

# ----------------------------------------------------------
# Plot training and validation accuracy per epoch
# ----------------------------------------------------------
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

# ----------------------------------------------------------
# Plot training and validation loss per epoch
# ----------------------------------------------------------
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.title('Training and validation loss')
plt.legend()
plt.figure()

# ----------------------------------------------------------
# Save results to a text file
# ----------------------------------------------------------
resultados_path = os.path.join(SCRIPT_DIR, 'resultado.txt')
model_summary_string = []
model.summary(print_fn=lambda x: model_summary_string.append(x))
model_summary_text = "\n".join(model_summary_string)

with open(resultados_path, 'w', encoding='utf-8') as f:
    f.write("=== RESULTADOS DO TREINAMENTO (TRANSFER LEARNING) ===\n\n")
    f.write(f"Diretório Base: {base_dir}\n")
    f.write(f"Total de Imagens de Gatos (Origem): {len(os.listdir(CAT_SOURCE_DIR))}\n")
    f.write(f"Total de Imagens de Cães (Origem): {len(os.listdir(DOG_SOURCE_DIR))}\n\n")
    
    f.write("Divisão dos Dados:\n")
    f.write(f"- Treinamento (Gatos): {len(os.listdir(TRAINING_CATS_DIR))}\n")
    f.write(f"- Treinamento (Cães): {len(os.listdir(TRAINING_DOGS_DIR))}\n")
    f.write(f"- Teste (Gatos): {len(os.listdir(TESTING_CATS_DIR))}\n")
    f.write(f"- Teste (Cães): {len(os.listdir(TESTING_DOGS_DIR))}\n\n")
    
    f.write("=== ARQUITETURA DO MODELO ===\n")
    f.write(model_summary_text)
    f.write("\n\n")
    
    f.write("=== HISTÓRICO DE TREINAMENTO POR ÉPOCA ===\n")
    f.write(f"{'Época':<8}{'Perda (Loss)':<15}{'Acurácia (Acc)':<18}{'Perda Val (Val Loss)':<22}{'Acurácia Val (Val Acc)'}\n")
    f.write("-" * 75 + "\n")
    for i in range(len(acc)):
        f.write(f"{i+1:<8}{loss[i]:<15.4f}{acc[i]:<18.4f}{val_loss[i]:<22.4f}{val_acc[i]:.4f}\n")
    f.write("\n")
    f.write("=== MÉTRICAS FINAIS ===\n")
    f.write(f"Melhor Acurácia de Treino: {max(acc):.4f} na época {acc.index(max(acc))+1}\n")
    f.write(f"Melhor Acurácia de Validação: {max(val_acc):.4f} na época {val_acc.index(max(val_acc))+1}\n")
    f.write(f"Menor Perda de Treino: {min(loss):.4f} na época {loss.index(min(loss))+1}\n")
    f.write(f"Menor Perda de Validação: {min(val_loss):.4f} na época {val_loss.index(min(val_loss))+1}\n")

print(f"\nResultados salvos com sucesso em: {resultados_path}")

# Desired output. Charts with training and validation metrics. No crash :)