


import tensorflow as tf
import matplotlib.pyplot as plot
import numpy as np

from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.engine import training

img_size = 100

def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds = image_dataset_from_directory(
    directory='fruits-360/Training',
    validation_split=0.2,
    subset="training",
    labels='inferred', # group name from folder will refer to 0,1,2,3,... so Apple Braeburn - 0
    label_mode='categorical',
    color_mode="rgb",
    batch_size=32, #size of group that is presented to network at once
    image_size=(img_size, img_size),
    shuffle=True,
    seed=123
)

valid_ds = image_dataset_from_directory(
    directory='fruits-360/Training',
    validation_split=0.2,
    subset="validation",
    labels='inferred', # group name from folder will refer to 0,1,2,3,... so Apple Braeburn - 0
    label_mode='categorical',
    color_mode="rgb",
    batch_size=32, #size of group that is presented to network at once
    image_size=(img_size, img_size),
    shuffle=True,
    seed=123
)

test_ds = image_dataset_from_directory(
    directory='fruits-360/Test',
    labels='inferred', # group name from folder will refer to 0,1,2,3,... so Apple Braeburn - 0
    label_mode='categorical',
    color_mode="rgb",
    batch_size=32, #size of group that is presented to network at once
    image_size=(img_size, img_size),
    shuffle=True,
    seed=None
)

#test_multiple_ds = image_dataset_from_directory(
#    directory='fruits-360/test-multiple-fruits',
#    color_mode="rgb",
#    batch_size=32, #size of group that is presented to network at once
#    image_size=(100, 100),
#    shuffle=True,
#    seed=None
#)

class_names = train_ds.class_names
print(class_names)

plot.figure(figsize=(img_size, img_size))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plot.subplot(3, 3, i + 1)
    plot.imshow(images[i].numpy().astype("uint8"))
    plot.title(class_names[i].title())
    plot.axis("off")
plot.show()

train_ds = train_ds.map(process) # normalization


