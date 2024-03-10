# import tensorflow_hub as hub
# import tensorflow as tf
# model_gcs_path = "gs://tfhub-modules/sayakpaul/convnext_tiny_1k_224/1/uncompressed"
# model = tf.keras.models.load_model(model_gcs_path)
# print(model.summary(expand_nested=True))
#
# Add your code here
#
# from google.colab import drive

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# drive.mount('/content/drive')
# paths=['/content/drive/MyDrive/train_val']


# set this to the path where you unzip the Google Driver zip download
# in bash shell in the current working directory type `unzip train_val-20240219T160046Z-001.zip && rm train_val-20240219T160046Z-001`
path='./train_val'

pixels = []
ages = []
genders = [] 

files = os.listdir(path)
i=0
for img in files:
  i=i+1
  gender = img.split("_")[1]
  age = img.split("_")[0]
  if gender != '0' and gender != '1':
    print(img)
  img = cv2.imread(str(path)+"/"+str(img))
  img=cv2.resize(img,(128,128))
  pixels.append(np.array(img))
  genders.append(np.array(gender))
  ages.append(np.array(age))
    # if i == 100:
    #   break
pixels = np.array(pixels,np.float32)
genders = np.array(genders,np.int64)
ages = np.array(ages,np.int64)

#
# Add your code here
#
from sklearn.model_selection import train_test_split

(x_train, x_test,
 y_train_gender, y_test_gender,
 y_train_age, y_test_age) = train_test_split(pixels,genders,ages,random_state=100,train_size=0.80, stratify=genders)

x_train=x_train/255
x_test=x_test/255





# Import dependencies
import tensorflow as tf
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,BatchNormalization,Flatten,Input,AveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input,Activation,GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB7

input_shape = (128, 128, 3)
inputs = Input(shape=input_shape)

modelB = VGG19(include_top=False, 
                weights='imagenet', 
                # input_tensor=inputs,
                input_shape=input_shape, 
                pooling=None, 
                classes=2,
                classifier_activation='softmax'
)
modelB = VGG16(include_top=False, 
                weights='imagenet', 
                input_tensor=inputs,
                input_shape=input_shape, 
                pooling=None, 
                classes=2,
                classifier_activation='softmax',
                )

# modelB = EfficientNetB7(include_top=False, 
#                       weights='imagenet', 
#                       # input_tensor=inputs,
#                       input_shape=input_shape, 
#                       # pooling=None, 
#                       classes=1,
#                       classifier_activation='softmax'
# )

from tensorflow.keras.utils import plot_model
# plot_model(model_resnet50, show_shapes=True)
# https://datascience.stackexchange.com/questions/77579/how-to-reduce-overfitting-in-a-pre-trained-network
# test getting rid of
n_bottom=8
n_top=0
for layer in modelB.layers[:n_top]:
    layer.trainable = True
for layer in modelB.layers[n_top:-n_bottom]:
    layer.trainable = False
for layer in modelB.layers[-n_bottom:]:
    layer.trainable = True
for layer in modelB.layers:
    print(layer, layer.trainable)


num_epochs=200
import pickle
from tensorflow import keras
from tensorflow.keras.layers import AveragePooling2D

# x_train3=np.load('x_train3_75.npy')
model_folder='output/'
checkpoint_filepath = 'output/checkpoint.age_gender_B4.h5'

callback_early=keras.callbacks.EarlyStopping(monitor='val_gender_output_accuracy',patience=100)
checkpointer = ModelCheckpoint(checkpoint_filepath, monitor='val_gender_output_accuracy', verbose=1, save_best_only=True,
                               save_weights_only=False, mode='auto', save_freq='epoch')
callback_list=[checkpointer, callback_early]
modelB.compile(optimizer=Adam(),loss='sparse_categorical_crossentropy', metrics='accuracy')

headModel=modelB.output
# headModel=modelB.layers[80].output
headModel_gender=AveragePooling2D(pool_size=(2,2))(headModel)
headModel_gender=Flatten()(headModel_gender)
headModel_gender=Dense(2048,activation='relu',
                       kernel_initializer=keras.initializers.GlorotUniform(seed=100),
                       bias_initializer=keras.initializers.Zeros())(headModel_gender)
headModel_gender=Dropout(0.1)(headModel_gender)
headModel_gender=Dense(2048,activation='relu',
                       kernel_initializer=keras.initializers.GlorotUniform(seed=100),
                       bias_initializer=keras.initializers.Zeros())(headModel_gender)
headModel_gender=Dropout(0.1)(headModel_gender)
headModel_gender=Dense(1024,activation='relu',
                       kernel_initializer=keras.initializers.GlorotUniform(seed=100),
                       bias_initializer=keras.initializers.Zeros())(headModel_gender)
headModel_gender=Dropout(0.1)(headModel_gender)
headModel_gender=Dense(512,activation='relu',
                       kernel_initializer=keras.initializers.GlorotUniform(seed=100),
                       bias_initializer=keras.initializers.Zeros())(headModel_gender)
headModel_gender=Dropout(0.1)(headModel_gender)
headModel_gender=Dense(128,activation='relu',
                       kernel_initializer=keras.initializers.GlorotUniform(seed=100),
                       bias_initializer=keras.initializers.Zeros())(headModel_gender)
headModel_gender=Dropout(0.1)(headModel_gender)
headModel_gender=Dense(64,activation='relu',
                       kernel_initializer=keras.initializers.GlorotNormal(seed=100),
                       bias_initializer=keras.initializers.Zeros())(headModel_gender)
headModel_gender=Dropout(0.1)(headModel_gender)
headModel_gender=Dense(1)(headModel_gender)
headModel_gender=Activation("sigmoid", name="gender_output")(headModel_gender)


headModel_age=AveragePooling2D(pool_size=(4,4))(headModel)
headModel_age=Flatten()(headModel_age)
headModel_age=Dense(2048,activation='relu')(headModel_age)
headModel_age=Dropout(0.2)(headModel_age)
headModel_age=Dense(1024,activation='relu')(headModel_age)
headModel_age=Dropout(0.2)(headModel_age)
headModel_age=Dense(512,activation='relu')(headModel_age)
headModel_age=Dropout(0.2)(headModel_age)
headModel_age=Dense(64,activation='relu')(headModel_age)
headModel_age=Dropout(0.2)(headModel_age)
headModel_age=Dense(1)(headModel_age)
headModel_age=Activation("linear", name="age_output")(headModel_age)


# augementation inside ImageDataGenerator since you cannot add to top of pre-trained model
# TF complains the graph cannot trace back to `Input()`
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers 

from tensorflow.keras import layers 


img_augmentation_layers = [
    layers.RandomRotation(factor=0.08),
    layers.RandomTranslation(height_factor=0.09, width_factor=0.09),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.09),
]

def img_augmentation(images):
  for layer in img_augmentation_layers:
    images = layer(images)
  return images

class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):

    # TO SLOW to do one image at a time
    # def __init__(self, color=False, **kwargs):
    #     super().__init__(preprocessing_function=self.preprocess, **kwargs)

    # def preprocess(self, image):
    #     # image = self.augment_color(image)
    #     for layer in img_augmentation_layers:
    #         image = layer(image)
    #     return image

    '''
    Way to go GOOGLE.  Keep up the "great" work.  Ugh. :( https://github.com/keras-team/keras/issues/12639
    And also why deprecate `ImageDataGenerator`, as making folder structures is more of a PAIN!!!!!
    '''
    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):
        
        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)


        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,
                                         shuffle=shuffle):
            flowx=img_augmentation(flowx) # faster to do this on a flow of a batch then each image indiv in a `preprocessing_function`
            target_dict = {}
            i = 0

            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict

datagen = MultiOutputDataGenerator()
datagen.fit(x_train)


# TRAIN
modelB_ = Model(inputs=modelB.input, outputs=[headModel_gender, headModel_age])
modelB_.compile(optimizer=Adam(learning_rate=0.00025,amsgrad=True),
               loss={'gender_output':'binary_crossentropy',
                     'age_output':'mse'},
               metrics={'gender_output':'accuracy',
                        'age_output':'mae'})
modelB_.fit(datagen.flow(x_train, y={"gender_output": y_train_gender.reshape((-1,1)), "age_output": y_train_age.reshape((-1,1))}, batch_size=8),
            steps_per_epoch=500,
           validation_data=(x_test, {"gender_output": y_test_gender, "age_output": y_test_age}),
           epochs=300, callbacks=[callback_list], validation_batch_size=8)

history2 = modelB_.history.history

with open('output/checkpoint.age_gender_B4.dict', 'wb') as file_pi:
    pickle.dump(history2, file_pi)