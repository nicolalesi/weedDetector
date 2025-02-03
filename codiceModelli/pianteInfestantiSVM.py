import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
import joblib

# Define dataset paths for training and test sets
train_dataset_path = 'dataset/WeedCrop.v1i.yolov5pytorch/train/images'
train_labels_path = 'dataset/WeedCrop.v1i.yolov5pytorch/train/labels'
test_dataset_path = 'dataset/WeedCrop.v1i.yolov5pytorch/test/images'
test_labels_path = 'dataset/WeedCrop.v1i.yolov5pytorch/test/labels'

# Load training images and labels
train_images = []
train_labels = []

for file_name in os.listdir(train_dataset_path):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Image formats
        image_path = os.path.join(train_dataset_path, file_name)
        label_path = os.path.join(train_labels_path, os.path.splitext(file_name)[0] + '.txt')

        if os.path.exists(label_path):
            # Read the image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))  # Resize for consistency

            # Read the label file and extract the class (first number)
            with open(label_path, 'r') as label_file:
                label_data = label_file.readlines()
                if label_data:  # If there are labels in the file
                    class_id = int(label_data[0].split()[0])  # The first number in the label file is the class
                    train_labels.append(class_id)
                    # Extract HOG features
                    features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                    train_images.append(features)

# Convert to NumPy arrays
X_train = np.array(train_images)
y_train = np.array(train_labels)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Load test images and labels
test_images = []
test_labels = []

for file_name in os.listdir(test_dataset_path):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Image formats
        image_path = os.path.join(test_dataset_path, file_name)
        label_path = os.path.join(test_labels_path, os.path.splitext(file_name)[0] + '.txt')

        if os.path.exists(label_path):
            # Read the image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))  # Resize for consistency

            # Read the label file and extract the class (first number)
            with open(label_path, 'r') as label_file:
                label_data = label_file.readlines()
                if label_data:  # If there are labels in the file
                    class_id = int(label_data[0].split()[0])  # The first number in the label file is the class
                    test_labels.append(class_id)
                    # Extract HOG features
                    features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                    test_images.append(features)

# Convert to NumPy arrays
X_test = np.array(test_images)
y_test = np.array(test_labels)

# Encode labels (using the same LabelEncoder as for the training set)
y_test = le.transform(y_test)

# Check the classes in train and test sets
print("Classi nel training set:", np.unique(y_train))
print("Classi nel test set:", np.unique(y_test))

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Predict on test set
y_pred = svm_model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(svm_model, 'modelliGenerati\svm_plant_classifier.pkl')

from sklearn.metrics import accuracy_score, classification_report

# Calcola le metriche
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Salva le metriche su un file di testo
with open('SVMMetriche.txt', 'w') as f:
    f.write("Accuracy: " + str(accuracy) + "\n\n")
    f.write("Classification Report:\n" + class_report)
