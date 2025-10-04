import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define the class names
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# 2. Load and preprocess a single image
test_image_path = "C:/Users/khush/Desktop/datasets/Data/Non Demented/OAS1_0001_MR1_mpr-1_152.jpg"
img = Image.open(test_image_path).convert('RGB')
img = img.resize((176, 176))
img_array = np.array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

# 3. Make a prediction
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

# 4. Print the results
print(f"The model predicts the image is: {predicted_class}")
print(f"Confidence score: {confidence:.2f}%")