#
# Add your code here
#
# from google.colab import drive

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
# drive.mount('/content/drive')
# paths=['/content/drive/MyDrive/train_val']


# set this to the path where you unzip the Google Driver zip download
# in bash shell in the current working directory type `unzip train_val-20240219T160046Z-001.zip && rm train_val-20240219T160046Z-001`
path='./train_val'

images = []
ages = []
genders = [] 

files = os.listdir(path)
i=0
for file in files:
  i=i+1
  age = file.split("_")[0]
  gender = file.split("_")[1]
  if i % 500 == 0:
    print("File: %s, Age: %s, Gender: %s" %(file, age, gender))
  # if gender != '0' and gender != '1':
  #   print(img)
  img = cv2.imread(path+"/"+file)
  img = cv2.resize(img,(128,128))
  images.append(np.array(img))
  genders.append(np.array(gender))
  ages.append(np.array(age))
    # if i == 100:
    #   break
images = np.array(images,np.float32)
genders = np.array(genders,np.int64)
ages = np.array(ages,np.int64)
images.shape


#
# Add your code here
#
from sklearn.model_selection import train_test_split

(x_train, x_test,
 y_train_gender, y_test_gender,
 y_train_age, y_test_age) = train_test_split(*[images,genders,ages],random_state=100,train_size=0.80, stratify=genders)

# x_train=x_train/255
# x_test=x_test/255
print(x_train.shape, x_test.shape, np.unique(y_train_gender, return_counts=True))



from tensorflow.keras import layers 

img_augmentation_layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.2),
]


def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images


x_train=x_train/255
x_test=x_test/255

# x_train2 = [img_augmentation(i) for i in x_train]
# np.save('x_train2.npy',x_train2)

# augment every 25th image
# x_train3 = []
# for i, image in enumerate(x_train):
#     if i % 20 == 0:
#         print(i)
#         x_train3.append(img_augmentation(image))
#     else:
#         x_train3.append(image)
# np.save('x_train3_80.npy', x_train3)
# x_train3=np.load('x_train3_80.npy')

# x_test2 = [img_augmentation(i) for i in x_test]
# np.save('x_test2.npy', x_test2)
# x_train2=np.load('x_train2.npy')
# x_train2.shape
# we can test augmenting later but not sure we want/need to


# Import dependencies
import tensorflow as tf
from tensorflow import keras
import pandas as pd
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input,Activation
from tensorflow.keras import layers 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# input_shape = (128, 128, 3)
# inputs = Input(shape=input_shape)
# hex(id(inputs)) # learned that graph in tf needs to use the same input memory location for both branches


def make_upstream(inputs, no_of_conv_layers=(16,32)):
    '''
    This is the "tail" if using language of queue data-structure or 
    "top" if using language of a stack data-structure
    of the model
    '''
    x=inputs
    x=layers.RandomRotation(factor=0.05)(x)
    x=layers.RandomTranslation(height_factor=0.048, width_factor=0.05)(x)
    # x=layers.RandomFlip()(x)
    x=layers.RandomContrast(factor=0.048)(x)

    for filters in no_of_conv_layers:
        print("Number filters per Conv2d layer:%s" % filters)
        x=Conv2D(filters=filters, kernel_size=(3,3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.0001))(x)
        x=Activation('relu')(x)
        x=BatchNormalization(axis=-1)(x)
        x=MaxPooling2D(pool_size = (2,2))(x)
        x=Dropout(0.15)(x) # increase from 0.1 to decrease overfitting

    return x


def build_age_branch(inputs, no_of_conv_layers=(16,32)):
    '''
    This is the age_branch of the model, bottom if queue or "header" if stack
    '''
    x=make_upstream(inputs, no_of_conv_layers)
    x=Flatten()(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(1)(x)
    x=Activation("linear", name="age_output")(x)
    
    return x


def build_gender_branch(inputs, no_of_conv_layers=(16,32)):
    '''
    This is the gender_branch of the model, bottom if queue or "header" if stack
    '''
    x=make_upstream(inputs, no_of_conv_layers)
    x=Flatten()(x)
    x=Dense(256,activation='relu')(x)
    x=Dropout(0.1)(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(32,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(1)(x)
    x=Activation("sigmoid", name="gender_output")(x)
    
    return x

input_shape = (128, 128, 3)
inputs = Input(shape=input_shape)
age_branch = build_age_branch(inputs, no_of_conv_layers=(16,32,64,128,512)) # inputs needs to be same memory address
gender_branch = build_gender_branch(inputs, no_of_conv_layers=(16,32,64,128,512))

modelA = Model(inputs=inputs,
               outputs = [gender_branch, age_branch],
               name="faces")
# modelA.summary()
from tensorflow.keras.utils import plot_model
# plot_model(modelA, show_shapes=True)


#
# Add your code here
#-
# modelA = Model(inputs=inputs,
#                outputs = [gender_branch,age_branch],
#                name="faces")
num_epochs=300
model_folder='output/'
checkpoint_filepath = 'output/checkpoint.age_gender_A4.h5'
checkpointer = ModelCheckpoint(checkpoint_filepath, monitor='val_gender_output_accuracy', verbose=1, save_best_only=True,
                               save_weights_only=False, mode='auto', save_freq='epoch')
callback_early=keras.callbacks.EarlyStopping(monitor='val_gender_output_accuracy',patience=50)

callback_list=[checkpointer, callback_early]

modelA.compile(optimizer=Adam(learning_rate=0.0012),
               loss={'gender_output':'binary_crossentropy',
                     'age_output':'mse'},
               metrics={'gender_output':'accuracy',
                        'age_output':'mae'})
modelA.fit(x=x_train,
           y={"gender_output": y_train_gender, "age_output": y_train_age},
           validation_data=(x_test,{"gender_output": y_test_gender, "age_output": y_test_age}),
           epochs=num_epochs, callbacks=[callback_list], batch_size=4, validation_batch_size=4)
    

history1 = modelA.history.history
# modelA.save(model_folder+"age_gender_A.h5")
# checkpoint.age_gender_A3 = val_gender_output_accuracy: 0.91300 - val_age_output_mae: 6.67
with open('output/checkpoint.age_gender_A4.dict', 'wb') as file_pi:
    pickle.dump(history1, file_pi)