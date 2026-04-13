# -*- coding: utf-8 -*-
"""cat_dog_classification_CNN.py
Run this in Google Colab"""

!pip install tensorflow_datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,BatchNormalization
from tensorflow.keras.models import Sequential
import os
import tensorflow_datasets as tfds
import tensorflow as tf

"""**Data Set Information**"""

dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

class_names = info.features['label'].names
class_names

#creating a directory
for i, example in enumerate(dataset['train']):
  # example = (image, label)
  image, label = example
  save_dir = './cats_vs_dogs/train/{}'.format(class_names[label])
  os.makedirs(save_dir, exist_ok=True)

  filename = save_dir + "/" + "{}_{}.jpg".format(class_names[label], i)
  tf.keras.preprocessing.image.save_img(filename, image.numpy())
  # print(filename)
  # break

import cv2
cat = cv2.imread('/content/cats_vs_dogs/train/cat/cat_10026.jpg')
cat

dog = cv2.imread('/content/cats_vs_dogs/train/dog/dog_10020.jpg')
dog

from google.colab.patches import cv2_imshow
cv2_imshow(dog)

from google.colab.patches import cv2_imshow
cv2_imshow(cat)

#set up the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2,rotation_range=10,
                             width_shift_range=0.1,height_shift_range=0.1,
                             shear_range=0.1,zoom_range=0.1,horizontal_flip=True)

datagen

train_generator = datagen.flow_from_directory('/content/cats_vs_dogs/train',target_size=(150,150),batch_size=32,
                                              class_mode='binary',subset='training')

validation_generator = datagen.flow_from_directory('/content/cats_vs_dogs/train',target_size = (150,150),batch_size=32,
                                                   class_mode='binary',subset='validation')

"""**Training-Epoch means training the machine with images and later testing with a untrained data**"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Model definition
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping and training
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
training_history = model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[early_stop])
print(training_history.history)

#save model
model.save('cats_vs_dogs.keras')

model_load=tf.keras.models.load_model('cats_vs_dogs.keras')

import requests
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# 🔹 Provide a direct image URL
img_url = "https://headsupfortails.com/cdn/shop/articles/Banner_0349af78-55fb-4814-9fe7-c08abe2bdbf4.jpg?v=1755865772"

# 🔹 Load and resize image
img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((150, 150))

# 🔹 Convert to array
img_array = image.img_to_array(img)

# 🔹 Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# 🔹 Normalize (FIXED: normalize img_array, not img)
img_array = img_array / 255.0

# 🔹 Predict (FIXED: use img_array)
prediction = model.predict(img_array)

# 🔹 Binary classification
TH = 0.5
prediction = int(prediction[0][0] > TH)

# 🔹 Class labels
Classes = {v: k for k, v in train_generator.class_indices.items()}

print(Classes[prediction])

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# 🔹 Provide a direct image URL
img_url = "https://www.nylabone.com/-/media/project/oneweb/nylabone/images/dog101/10-intelligent-dog-breeds/golden-retriever-tongue-out.jpg?h=430&w=710&hash=7FEB820D235A44B76B271060E03572C7"

# 🔹 Load and resize image
img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((150, 150))

# 🔹 Convert to array
img_array = image.img_to_array(img)

# 🔹 Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# 🔹 Normalize (FIXED: normalize img_array, not img)
img_array = img_array / 255.0

# 🔹 Predict (FIXED: use img_array)
prediction = model.predict(img_array)

# 🔹 Binary classification
TH = 0.5
prediction = int(prediction[0][0] > TH)

# 🔹 Class labels
Classes = {v: k for k, v in train_generator.class_indices.items()}

print(Classes[prediction])
