import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Load your trained model
model = tf.keras.models.load_model('./modelliGenerati/plant_disease_model_inception.h5')  # Replace 'your_model_directory' with the path to your saved model

# Load and preprocess your image
img_path = './A_sunflower.jpg'  # Replace 'path_to_your_image.jpg' with your image file path
img = image.load_img(img_path, target_size=(139, 139))  # Resize to match the input size of your model

# Load and preprocess your image
img_path = './A_sunflower.jpg'  # Replace with the path to your image file
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(139, 139))  # Load the image and resize
img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert image to array
img_array = tf.image.resize(img_array, (139, 139))  # Resize the image to match the model's input size
img_array = tf.expand_dims(img_array, axis=0)  # Add a batch dimension

# Get the predictions for the image
predictions = model.predict(img_array)
predicted_class = tf.argmax(predictions[0])
print("CLASSE PREDETTA ",predicted_class)
# Get the predictions for the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(predicted_class)

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
