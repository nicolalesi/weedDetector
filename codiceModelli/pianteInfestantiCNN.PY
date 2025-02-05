import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical  # Aggiunto import
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset paths
dataset_path = '../dataset/WeedCrop.v1i.yolov5pytorch'
train_path = os.path.join(dataset_path, 'train')
valid_path = os.path.join(dataset_path, 'valid')
test_path = os.path.join(dataset_path, 'test')

# Model parameters
batch_size = 128
num_epochs = 10
image_size = (139, 139)
num_classes = 2
learning_rate = 0.001

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

# Bilanciamento delle classi: calcolare i pesi delle classi
class_weights = {0: 1., 1: 1.}  # Default weights (uguali)
train_labels = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
class_counts = {0: 0, 1: 0}

# Calcolare il numero di immagini per ciascuna classe nel training set
for label in train_labels:
    class_counts[0] += len([f for f in os.listdir(os.path.join(train_path, label))])
    class_counts[1] += len([f for f in os.listdir(os.path.join(train_path, label))])

total_images = sum(class_counts.values())
class_weights[0] = total_images / (2 * class_counts[0])  # Peso per la classe 0
class_weights[1] = total_images / (2 * class_counts[1])  # Peso per la classe 1

# Data generators
train_dataset = train_datagen.flow_from_directory(
    train_path, target_size=image_size, batch_size=batch_size, class_mode='categorical'
)
valid_dataset = valid_datagen.flow_from_directory(
    valid_path, target_size=image_size, batch_size=batch_size, class_mode='categorical'
)
test_dataset = test_datagen.flow_from_directory(
    test_path, target_size=image_size, batch_size=batch_size, class_mode='categorical'
)

# Load pre-trained InceptionV3 model
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
    train_dataset,
    epochs=num_epochs,
    validation_data=valid_dataset,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save model
model.save('../modelliGenerati/plant_disease_model_inception.h5')

# Evaluate model
test_labels = to_categorical(test_dataset.classes, num_classes=num_classes)  # Codifica one-hot delle etichette di test
y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_labels, axis=1)

# Calcolo metriche
accuracy = accuracy_score(y_true, y_pred_classes)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
class_report = classification_report(y_true, y_pred_classes, digits=4, output_dict=True)

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
