from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Input
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
from sklearn.metrics import classification_report
import time 

import os

directory = './dataset/WeedCrop.v1i.yolov5pytorch/train'
contents = os.listdir(directory)
num_of_dirs = len([name for name in contents if os.path.isdir(os.path.join(directory, name))])

print("Contents of the directory:")
for item in contents:
    print(item)

print(f"\nNumber of directories: {num_of_dirs}")

from PIL import Image
import os

# Define the directory path
directory_path = 'dataset/WeedCrop.v1i.yolov5pytorch/train'

# List all files in the directory
file_names = os.listdir(directory_path)

# Load images from the directory
images = []
for file_name in file_names:
    if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
        image_path = os.path.join(directory_path, file_name)
        image = Image.open(image_path)
        images.append(image)

# Process the images as required
# ...

# Example: Showing the first image
if images:
    images[0].show()
else:
    print("No images found in the directory.")

# Define parameters
batch_size = 128
num_epochs = 10
image_size = (139, 139)
num_classes = 2

# Load the InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*image_size, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
class_outputs = Dense(num_classes, activation='softmax')(x)


# Create the model
model = Model(inputs=base_model.input, outputs=class_outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

# Load the training data with aggressive data augmentation
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

train_dataset = train_datagen.flow_from_directory(
    'dataset/WeedCrop.v1i.yolov5pytorch/train',
    target_size=(139, 139),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the validation data with moderate data augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_dataset = val_datagen.flow_from_directory(
    'dataset/WeedCrop.v1i.yolov5pytorch/valid',
    target_size=(139, 139),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define learning rate scheduler
def lr_scheduler(epoch):
    if epoch < 10:
        return 0.001
    elif 10 <= epoch < 20:
        return 0.0001
    else:
        return 0.00001

lr_schedule = LearningRateScheduler(lr_scheduler)

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define model checkpoint to save the best model
checkpoint = ModelCheckpoint('modelliGenerati/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
# Define ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[lr_schedule, early_stop, checkpoint, reduce_lr]
)

# Save the model in native Keras format
#model.save('plant_disease_model_inception.keras')

#import joblib

# Save the model using joblib
#joblib.dump(model, 'plant_disease_model_inception.pkl')

# Plot the metrics to visualize the training process
import matplotlib.pyplot as plt

def plot_metrics(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

plot_metrics(history)

# Evaluate the model on the test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_dataset = test_datagen.flow_from_directory(
    'dataset/WeedCrop.v1i.yolov5pytorch/test',
    target_size=(139, 139),
    batch_size=batch_size,
    class_mode='categorical'
)

test_labels = test_dataset.classes
test_labels = to_categorical(test_labels, num_classes=num_classes)

start_time = time.time()
y_pred = model.predict(test_dataset)
y_pred_bool = np.argmax(y_pred, axis=1)
rounded_labels = np.argmax(test_labels, axis=1)

print(classification_report(y_pred_bool, rounded_labels, digits=4))
print("Time taken to predict the model: " + str(time.time() - start_time))

# Save the model
model.save('modelliGenerati/plant_disease_model_inception.h5')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Load your trained model
model = tf.keras.models.load_model('modelliGenerati/plant_disease_model_inception.h5')  # Replace 'your_model_directory' with the path to your saved model

# Load and preprocess your image
img_path = 'dataset/pianteInfestanti/test/ridderzuring_3126_jpg.rf.8980b3ae3ec4ecd023aab5bc54c26089.jpg'  # Replace 'path_to_your_image.jpg' with your image file path
img = image.load_img(img_path, target_size=(139, 139))  # Resize to match the input size of your model

# Load and preprocess your image
img_path = 'dataset/WeedCrop.v1i.yolov5pytorch/test/images/12122_jpg.rf.d7313af9bab0e80b7149ee0a3c32caf8.jpg'  # Replace with the path to your image file
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(139, 139))  # Load the image and resize
img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert image to array
img_array = tf.image.resize(img_array, (139, 139))  # Resize the image to match the model's input size
img_array = tf.expand_dims(img_array, axis=0)  # Add a batch dimension

# Get the predictions for the image
predictions = model.predict(img_array)
predicted_class = tf.argmax(predictions[0])

# Get the predictions for the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Generate the heatmap
last_conv_layer = model.get_layer('mixed10')  
heatmap_model = tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = heatmap_model(img_array)
    loss = predictions[:, predicted_class]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
heatmap = np.maximum(heatmap, 0)

heatmap_resized = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

# Convert both arrays to the same data type (e.g., unsigned 8-bit integer)
img_array_uint8 = (img_array[0].numpy() * 255).astype(np.uint8)
heatmap_resized_uint8 = (heatmap_resized * 255).astype(np.uint8)  # Adjust the range of heatmap values

# Overlay the heatmap on the original image
heatmap_resized_uint8 = cv2.applyColorMap(heatmap_resized_uint8, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img_array_uint8, 0.6, heatmap_resized_uint8, 0.4, 0)

# Display the original image, heatmap, and overlay
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(img)
plt.title('Original Image')

plt.subplot(132)
plt.imshow(heatmap_resized_uint8)
plt.title('Heatmap')

plt.subplot(133)
plt.imshow(superimposed_img)
plt.title('Overlay')

plt.tight_layout()
plt.show()