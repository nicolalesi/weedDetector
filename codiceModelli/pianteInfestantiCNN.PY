import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical  # Aggiunto import
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset paths
dataset_path = '../dataset/WeedCrop.v1i.yolov5pytorch'
train_path = os.path.join(dataset_path, 'train/images')
train_label_path = os.path.join(dataset_path, 'train/labels')
valid_path = os.path.join(dataset_path, 'valid/images')
valid_label_path = os.path.join(dataset_path, 'valid/labels')
test_path = os.path.join(dataset_path, 'test/images')
test_label_path = os.path.join(dataset_path, 'test/labels')

# Model parameters
batch_size = 128
num_epochs = 10
image_size = (139, 139)
num_classes = 2
learning_rate = 0.001

# Funzione per caricare i dati YOLO
def load_yolo_data(images_path, labels_path, image_size):
    images = []
    labels = []
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Numero di immagini trovate: {len(image_files)}")

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
                    class_id = int(label_data[0].split()[0])  # Il primo numero è la classe
                    labels.append(class_id)
                else:
                    print(f"Etichetta vuota per l'immagine: {image_file}")
                    labels.append(0)  # Assegna una classe predefinita (ad esempio, 0)
        else:
            print(f"Etichetta mancante per l'immagine: {image_file}")
            labels.append(0)  # Assegna una classe predefinita (ad esempio, 0)

    print(f"Numero di etichette caricate: {len(labels)}")
    
    return np.array(images), np.array(labels)


# Carica i dati di addestramento
X_train, y_train = load_yolo_data(train_path, train_label_path, image_size)
X_valid, y_valid = load_yolo_data(valid_path, valid_label_path, image_size)
X_test, y_test = load_yolo_data(test_path, test_label_path, image_size)

# Codifica one-hot delle etichette
y_train = to_categorical(y_train, num_classes=num_classes)
y_valid = to_categorical(y_valid, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Bilanciamento delle classi: calcolare i pesi delle classi
class_weights = {0: 1., 1: 1.}  # Default weights (uguali)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Model setup
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
for layer in base_model.layers:
    layer.trainable = False

# Custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile model with class weights
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate),
    metrics=['accuracy']
)

# Callbacks
def lr_scheduler(epoch):
    return learning_rate * (0.1 ** (epoch // 10))

callbacks = [
    LearningRateScheduler(lr_scheduler),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('../modelliGenerati/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

# Train model with class weights
history = model.fit(
    X_train, y_train,
    epochs=num_epochs,
    validation_data=(X_valid, y_valid),
    class_weight=class_weights,
    batch_size=batch_size,
    callbacks=callbacks
)

# Save model
model.save('../modelliGenerati/plant_disease_model_inception.h5')

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(np.unique(y_test, axis=0))  # Controlla che le etichette siano distribuite correttamente
class_counts = np.unique(y_test, return_counts=True)
print("Distribuzione delle classi nel test set:", class_counts)
print("Prime 10 predizioni softmax:", y_pred[:10])

# Calcolo metriche
accuracy = accuracy_score(y_true, y_pred_classes)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
class_report = classification_report(y_true, y_pred_classes, digits=4)

# Stampiamo i risultati
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", class_report)
print("\nConfusion Matrix:\n", conf_matrix)

# Salviamo le metriche in un file
with open('../MetricheModelli/CNNMetrich.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n" + classification_report(y_true, y_pred_classes, digits=4) + "\n\n")
    f.write("Confusion Matrix:\n" + str(conf_matrix) + "\n")

# Plot metrics
def plot_metrics(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy') 
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.show()

plot_metrics(history)
