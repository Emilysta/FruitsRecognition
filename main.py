


import tensorflow as tf
import matplotlib.pyplot as plot
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

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
test, test_info = test_ds.take(1)
fig= tf.data.Dataset.show_examples(test,test_info)
fig.show()

#test_multiple_ds = image_dataset_from_directory(
#    directory='fruits-360/test-multiple-fruits',
#    color_mode="rgb",
#    batch_size=32, #size of group that is presented to network at once
#    image_size=(100, 100),
#    shuffle=True,
#    seed=None
#)

class_names = train_ds.class_names
#classes = train_ds.classes
#print(classes)
print(class_names)
print(len(class_names))

if(os.stat("model.h5").st_size==0) : # construct model of CNN
    model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16,3,activation = 'relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32,3,activation = 'relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64,3,activation = 'relu'),
    layers.MaxPool2D(),
    layers.Conv2D(128,3,activation = 'relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(1024,activation='relu'),
    layers.Dense(512,activation='relu'),
    layers.Dense(len(class_names),activation = 'softmax')
    ])
    model.compile(loss = 'categorical_crossentropy',
                optimizer='adam', metrics =['accuracy']) 
    history =model.fit(
    train_ds,
    validation_data = valid_ds,
    epochs = 3
    )
    model.summary()   
    model.save("model.h5")    #save trained model to file 
else :
    model = load_model('model.h5')
    
    predictions=model.predict(test_ds)
    predict, predict_info = predictions.take(1)

    fig= tf.data.Dataset.show_examples(predict,predict_info)
    fig.show()
    #i=0
    
    #plot.figure(figsize = (30, 30))
    #for images,  in predictions.take(1):
    #    image, label = images["image"],images["label"]
    #    plot.subplot(9,5, i + 1)
    #    plot.xlabel(label.numpy())
    #    plot.imshow(image.numpy()[:,:,0].astype(np.float32))
    #    i=i+1
    #plot.show()