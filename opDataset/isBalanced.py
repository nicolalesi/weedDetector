import os
import numpy as np
import matplotlib.pyplot as plt

# Directory del dataset di training
train_dataset_path = '../dataset/WeedCrop.v1i.yolov5pytorch/test/images'
train_labels_path = '../dataset/WeedCrop.v1i.yolov5pytorch/test/labels'

# Lista per raccogliere le etichette delle classi
train_labels = []

# Carica le etichette del training set
for file_name in os.listdir(train_dataset_path):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Formati immagine
        label_path = os.path.join(train_labels_path, os.path.splitext(file_name)[0] + '.txt')
        
        if os.path.exists(label_path):
            # Leggi il file delle etichette e estrai la classe (primo numero)
            with open(label_path, 'r') as label_file:
                label_data = label_file.readlines()
                if label_data:  # Se ci sono etichette nel file
                    class_id = int(label_data[0].split()[0])  # La prima cifra nel file è la classe
                    train_labels.append(class_id)

# Converti in un array NumPy per analisi
train_labels_array = np.array(train_labels)

# Calcola la distribuzione delle classi
class_counts = np.bincount(train_labels_array)

# Visualizza la distribuzione delle classi
class_names = [f"Classe {i}" for i in range(len(class_counts))]  # Cambia con i nomi reali delle classi
print("Distribuzione delle classi nel training set:")
for i, count in enumerate(class_counts):
    print(f"{class_names[i]}: {count} campioni")

# Visualizza la distribuzione in un grafico
plt.figure(figsize=(10, 6))
plt.bar(class_names, class_counts)
plt.xlabel('Classi')
plt.ylabel('Numero di campioni')
plt.title('Distribuzione delle classi nel training set')
plt.xticks(rotation=90)
plt.show()

