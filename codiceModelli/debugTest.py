import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Percorsi ai dataset e modello salvato
dataset_path = '../dataset/WeedCrop.v1i.yolov5pytorch'
test_dataset_path = os.path.join(dataset_path, 'test', 'images')  # Directory delle immagini
test_labels_path = os.path.join(dataset_path, 'test', 'labels')  # Directory delle etichette
model_path = '../modelliGenerati/plant_disease_model_inception.h5'

# Parametri del modello
image_size = (139, 139)
batch_size = 128
num_classes = 2  # Assumiamo due classi

# Caricamento del modello pre-addestrato
print("🔄 Caricamento del modello...")
model = tf.keras.models.load_model(model_path)
print("✅ Modello caricato con successo!")

# Caricare manualmente le immagini e le etichette YOLO
def load_yolo_data(images_path, labels_path, image_size):
    images = []
    labels = []
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        # Carica l'immagine
        img_path = os.path.join(images_path, image_file)
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0  # Normalizza l'immagine
        images.append(img_array)

        # Carica le etichette corrispondenti
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_path, label_file)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label_data = f.readlines()
                if label_data:
                    # Estrai la classe (primo numero) da ciascuna etichetta
                    class_id = int(label_data[0].split()[0])  # Il primo numero è la classe
                    labels.append(class_id)

    return np.array(images), np.array(labels)

# Carica i dati di test
print("🔄 Caricamento delle immagini e delle etichette di test...")
X_test, y_true = load_yolo_data(test_dataset_path, test_labels_path, image_size)
print(f"Numero di immagini di test caricate: {len(X_test)}")

# Previsione sulle immagini di test
print("🔄 Generazione delle predizioni...")
y_pred_prob = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=1)  # Convertire in etichette predette

# Analisi delle predizioni
accuracy = accuracy_score(y_true, y_pred_classes)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
class_report = classification_report(y_true, y_pred_classes, digits=4)

print("y_true",y_true)
# Stampare le metriche di valutazione
print(f"\n✅ Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:\n", class_report)
print("\n🧩 Confusion Matrix:\n", conf_matrix)

# Visualizzazione della matrice di confusione
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(5, 5))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

plot_confusion_matrix(conf_matrix)
