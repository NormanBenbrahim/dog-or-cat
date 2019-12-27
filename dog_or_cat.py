import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os 
import numpy as np 
import seaborn as sns
import pandas as pd 
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


##### define the data

# working with paths is a drag, but let's pickup our bootstraps like
# boomers keep telling us to
home_dir = os.path.expanduser("~")

if not os.path.exists(os.path.join(home_dir, '.keras/datasets/cats_and_dogs_filtered')):
    link = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', 
                                      origin=link, 
                                      extract=True)
    basedir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')

else:
    basedir = os.path.join(home_dir, '.keras/datasets/cats_and_dogs_filtered')

# now join the paths for training and testing, since they're already
# split for us
train_dir = os.path.join(basedir, 'train')
validate_dir = os.path.join(basedir, 'validation')

# they've split test & train by cats & dog subdirectories
train_cat_dir = os.path.join(train_dir, 'cats')
train_dog_dir = os.path.join(train_dir, 'dogs')

validate_cat_dir = os.path.join(validate_dir, 'cats')
validate_dog_dir = os.path.join(validate_dir, 'dogs')

##### initialize some variables
batch_size = 100
img_shape = (150, 150)

num_cats_tr = len(os.listdir(train_cat_dir))
num_dogs_tr = len(os.listdir(train_dog_dir))

num_cats_val = len(os.listdir(validate_cat_dir))
num_dogs_val = len(os.listdir(validate_dog_dir))
# all samples
total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val 


# the following will load in proper RGB array, convert into tensor
# and normalize [0, 1]. super useful

# first load 
train_img = ImageDataGenerator(rescale=1.0/255)
validate_img = ImageDataGenerator(rescale=1.0/255)

# now do the preprocessing 
# only need to shuffle the training data
train_data_gen = train_img.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=img_shape,
                                               class_mode='binary')

val_data_gen = validate_img.flow_from_directory(batch_size=batch_size,
                                                directory=validate_dir,
                                                shuffle=False,
                                                target_size=img_shape,
                                                class_mode='binary')



##### define the model (the heavy part)

# we will build 4 convolution blocks with a max pool layer 
# in each. the max pool layer will be the same for each layer
# the 3rd & 4th layers will be the same
model = tf.keras.models.Sequential([
    # convolution 1
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # convolution 2
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #convolution 3
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #convolution 4
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # flatten layer
    tf.keras.layers.Flatten(),
    # one super-dense layer, and one 2 neuron output layer (for 2 classes)
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])



#### compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#### train the model
# we need to use fit_generator instead of fit because of how we
# loaded the data
epochs = 100
steps_per_epoch = int(np.ceil(np.ceil(total_train)/np.float(batch_size)))
validation_steps = int(np.ceil(total_val/np.float(batch_size)))

##### HIGHLY RECOMMEND YOU RUN ON GOOGLE CLOUD/COLAB
##### VERY COMPUTATIONALLY HEAVY
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=validation_steps
    )