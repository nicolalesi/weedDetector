import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

# =====================
# Impostazioni e paths
# =====================
dataset_path = '/Users/mariocorrente/Desktop/df_aum/dsbase'
train_path = os.path.join(dataset_path, 'train/images')
train_label_path = os.path.join(dataset_path, 'train/labels')
valid_path = os.path.join(dataset_path, 'valid/images')
valid_label_path = os.path.join(dataset_path, 'valid/labels')
test_path = os.path.join(dataset_path, 'test/images')
test_label_path = os.path.join(dataset_path, 'test/labels')

# Modifica: riduciamo il learning rate per una migliore regolarizzazione
learning_rate = 1e-4  
batch_size = 128
num_epochs = 10
image_size = (139, 139)
num_classes = 2

# ==========================================
# Funzione per caricare i dati (YOLO format)
# ==========================================
def load_yolo_data(images_path, labels_path, image_size):
    print("Caricamento dati YOLO...")
    images = []
    labels = []
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Numero di immagini trovate: {len(image_files)}")
    for image_file in image_files:
        # Carica immagine
        img_path = os.path.join(images_path, image_file)
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0  # Normalizza l'immagine
        images.append(img_array)
        # Carica l'etichetta corrispondente
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_path, label_file)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label_data = f.readlines()
                if label_data:
                    class_id = int(label_data[0].split()[0])
                    labels.append(class_id)
                else:
                    print(f"Etichetta vuota per l'immagine: {image_file}")
                    labels.append(0)
        else:
            print(f"Etichetta mancante per l'immagine: {image_file}")
            labels.append(0)
    print(f"Numero di etichette caricate: {len(labels)}")
    return np.array(images), np.array(labels)

# ================================
# Caricamento dati: train, valid, test
# ================================
X_train, y_train = load_yolo_data(train_path, train_label_path, image_size)
X_valid, y_valid = load_yolo_data(valid_path, valid_label_path, image_size)
X_test, y_test = load_yolo_data(test_path, test_label_path, image_size)

# One-hot encode delle etichette
y_train = to_categorical(y_train, num_classes=num_classes)
y_valid = to_categorical(y_valid, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# =============================
# Combina training e validation
# =============================
X_train_combined = np.concatenate([X_train, X_valid])
y_train_combined = np.concatenate([y_train, y_valid])

# Class weights (bilanciati di default)
class_weights = {0: 1., 1: 1.}

# Data augmentation (puoi eventualmente usare train_datagen.flow(...) durante l'addestramento)
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

# ======================
# Funzione per costruire il modello
# ======================
def build_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
    # Manteniamo congelati tutti gli strati della base pre-addestrata
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Modifica: aggiunta di L2 regularization sul layer Dense
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    # Aumento del dropout per maggiore regolarizzazione
    x = Dropout(0.6)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model

# ====================
# Definizione dei callback (da usare in ogni fold)
# ====================
def lr_scheduler(epoch):
    return learning_rate * (0.1 ** (epoch // 10))

# Modifica: riduzione della patience per EarlyStopping e per ReduceLROnPlateau
callbacks_cv = [
    LearningRateScheduler(lr_scheduler),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
]

# ============================================
# Esecuzione della Cross Validation 5-fold
# ============================================
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
cv_accuracies = []

print("Inizio Cross Validation 5-Fold...")
for train_index, val_index in kf.split(X_train_combined):
    print(f"\n=== Fold {fold_no} ===")
    X_train_fold, X_val_fold = X_train_combined[train_index], X_train_combined[val_index]
    y_train_fold, y_val_fold = y_train_combined[train_index], y_train_combined[val_index]
    
    # Costruisci un nuovo modello per questo fold
    model_cv = build_model()
    
    # Allena il modello sul fold corrente
    history_cv = model_cv.fit(
        X_train_fold, y_train_fold,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(X_val_fold, y_val_fold),
        class_weight=class_weights,
        callbacks=callbacks_cv,
        verbose=1
    )
    
    # Valuta sul fold di validazione
    scores = model_cv.evaluate(X_val_fold, y_val_fold, verbose=0)
    print(f"Fold {fold_no} Accuracy: {scores[1]:.4f}")
    cv_accuracies.append(scores[1])
    fold_no += 1

print("\nAccuratezza media sui 5 fold: {:.4f}".format(np.mean(cv_accuracies)))

# ====================================================
# Addestramento del modello finale su tutto il training (train+valid)
# ====================================================
print("\nAddestramento del modello finale su tutto il training (train+valid)...")
final_model = build_model()
final_history = final_model.fit(
    X_train_combined, y_train_combined,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_split=0.1,  # usa una piccola porzione per monitorare il validation loss
    class_weight=class_weights,
    callbacks=callbacks_cv,
    verbose=1
)

# Salva il modello finale
final_model.save('/Users/mariocorrente/Desktop/df_aum/modelli/plant_disease_model_inception_final.h5')
print("Modello finale salvato.")

# ====================
# Valutazione sul test set
# ====================
print("\nValutazione del modello finale sul test set...")
y_pred = final_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Distribuzione delle etichette nel test set:", np.unique(y_test, axis=0))
class_counts = np.unique(y_test, return_counts=True)
print("Distribuzione delle classi nel test set:", class_counts)
print("Prime 10 predizioni softmax:", y_pred[:10])

accuracy = accuracy_score(y_true, y_pred_classes)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
class_report = classification_report(y_true, y_pred_classes, digits=4)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", class_report)
print("\nConfusion Matrix:\n", conf_matrix)

with open('/Users/mariocorrente/Desktop/df_aum/metriche/CNNMetrich.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n" + classification_report(y_true, y_pred_classes, digits=4) + "\n\n")
    f.write("Confusion Matrix:\n" + str(conf_matrix) + "\n")

# ====================
# Calcolo AUC-ROC (utilizzando la colonna 1, classe positiva, poiché y_test è one-hot encoded)
# ====================
print("Calcolo AUC-ROC...")
y_pred_final = final_model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred_final[:, 1])
roc_auc = auc(fpr, tpr)
print(f"AUC-ROC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# ====================
# Calcolo Precision, Recall, F1-Score
# ====================
print("Calcolo Precision, Recall, F1-Score...")
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# ====================
# Matrice di Confusione Normalizzata
# ====================
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# ====================
# Curve di Apprendimento (Learning Curves)
# ====================
def plot_learning_curves(history):
    print("Plotting Learning Curves...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_learning_curves(final_history)
