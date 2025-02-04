import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
import joblib
from imblearn.over_sampling import SMOTE

# Define dataset paths
train_dataset_path = '../dataset/WeedCrop.v1i.yolov5pytorch/train/images'
train_labels_path = '../dataset/WeedCrop.v1i.yolov5pytorch/train/labels'
test_dataset_path = '../dataset/WeedCrop.v1i.yolov5pytorch/test/images'
test_labels_path = '../dataset/WeedCrop.v1i.yolov5pytorch/test/labels'
valid_dataset_path = '../dataset/WeedCrop.v1i.yolov5pytorch/valid/images'
valid_labels_path = '../dataset/WeedCrop.v1i.yolov5pytorch/valid/labels'

# Funzione per estrarre le feature HOG
def extract_hog_features(img):
    return hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

# Funzione per caricare immagini e etichette
def load_dataset(image_path, label_path):
    images = []
    labels = []
    for file_name in os.listdir(image_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  
            img_path = os.path.join(image_path, file_name)
            lbl_path = os.path.join(label_path, os.path.splitext(file_name)[0] + '.txt')

            if os.path.exists(lbl_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128, 128))  

                with open(lbl_path, 'r') as f:
                    label_data = f.readlines()
                    if label_data:  
                        class_id = int(label_data[0].split()[0])  
                        labels.append(class_id)
                        features = extract_hog_features(img)
                        images.append(features)
    return np.array(images), np.array(labels)

# Carica dataset
X_train, y_train = load_dataset(train_dataset_path, train_labels_path)
X_test, y_test = load_dataset(test_dataset_path, test_labels_path)
X_val, y_val = load_dataset(valid_dataset_path, valid_labels_path)

# Codifica le etichette
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

# Controlla le classi nei dataset
print("Classi nel training set:", np.unique(y_train))
print("Classi nel test set:", np.unique(y_test))

# Gestione dello sbilanciamento con SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Distribuzione dopo SMOTE:", np.bincount(y_train_resampled))

# Inizializza il modello KNN con parametri fissi
knn_model = KNeighborsClassifier(n_neighbors=100, weights='distance', metric='euclidean')
knn_model.fit(X_train_resampled, y_train_resampled)

# Predizioni sul validation set
y_val_pred = knn_model.predict(X_val)

# Stampa i risultati sul validation set
print("Accuracy (Validation Set):", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred, target_names=[str(c) for c in le.classes_], zero_division=0))

# Predizioni sul test set
y_pred = knn_model.predict(X_test)

# Report di valutazione
target_names = [str(class_name) for class_name in le.classes_]
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Salva il modello KNN
joblib.dump(knn_model, '../modelliGenerati/knn_plant_classifier.pkl')

# Salva le metriche su un file
with open('../MetricheModelli/KNNMetriche.txt', 'w') as f:
    f.write("Accuracy: " + str(accuracy_score(y_test, y_pred)) + "\n\n")
    f.write("Classification Report:\n" + classification_report(y_test, y_pred))
