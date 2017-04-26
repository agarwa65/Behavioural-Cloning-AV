
# coding: utf-8

# In[1]:

import keras
from keras.models import Sequential
from keras.layers import Cropping2D, Dense, Dropout, Activation, Flatten, Convolution2D, Input, Lambda, SpatialDropout2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras import regularizers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import csv
import cv2
import os

from utils import *

PATH = "divya_data/"
IMG_PATH = "divya_data/IMG/"
DATA_PATH = os.path.join(PATH, "driving_log.csv")
BATCH_SIZE = 64
EPOCHS = 100
correction = 0.18
validation_split = 0.2

lines = []
images = []
measurements =[]


# In[2]:

#Function below describes the CNN model to be used for Behaviour Cloning
def nvidia_model():
    """Ref: Nvidia paper http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf"""
    
    def image_resize(img):
        
        import tensorflow as tf
        return tf.image.resize_images(img, (66,200))

    model = Sequential()
    # Cropping the input images
    model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
    # Resizing input images Output : 66x200
    # model.add(Lambda(image_resize))
    # Normalizing the input images
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    # Convolution Layer 1 : 24x3x3 filters, RELU activation, maxpooling(2,2) and dropout
    model.add(Convolution2D(24,3,3, activation='relu') )
    model.add(MaxPooling2D())
    #model.add(Dropout(0.24))
    # Convolution Layer 2 : 48x3x3 filters, RELU activation, maxpooling(2,2) and dropout
    model.add(Convolution2D(36,3,3, activation='relu'))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.24))
    # Convolution Layer 2 : 48x3x3 filters, RELU activation, maxpooling(2,2) and dropout
    model.add(Convolution2D(48,3,3, activation='relu'))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.24))
    # Convolution Layer 3 : 64x3x3 filters, RELU activation, maxpooling(2,2) and dropout
    model.add(Convolution2D(128,3,3, activation='relu'))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.24))
    # Flattening 
    model.add(Flatten())

    # Fully connected Layer: Output : 512 , RELU activation and dropout
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.52))
    # Fully connected Layer: Output : 256 , RELU activation and dropout
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.52))
    # Fully connected Layer: Output : 100 , RELU activation and dropout
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.52)) 
    
    #Output for Steering wheel angle control
    model.add(Dense(1))

    model.compile(optimizer='adam',loss ='mse')
    
    return model
    


# In[3]:

# Generator definition:
def image_data_generator(input_data, batch_size):

    while(1):
        shuffle(input_data)
        
        #for AWS instance running or adding current path in local instance
        for i in range(0, len(input_data), batch_size):
            
            images = []
            measurements = []
            input_batch = input_data[i:i+batch_size]
            
            for line in input_batch:
                sourcepath = line[0]
                filename = sourcepath.split('/')[-1]
                current_path = IMG_PATH + filename
                #print(current_path)
                image = mpimg.imread(current_path)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                measurement = float(line[3])
                measurements.append(measurement)

                ## Read center, left, right camera data and append it
                steering_center = measurement
                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                #read in images from center, left, right cameras
                left_sourcepath = line[1]
                left_filename = sourcepath.split('/')[-1]
                left_current_path = IMG_PATH + left_filename

                right_sourcepath = line[2]
                right_filename = sourcepath.split('/')[-1]
                right_current_path = IMG_PATH + right_filename

                # add images and angles to data set
                #images.append(cv2.cvtColor(cv2.imread(left_current_path),cv2.COLOR_BGR2RGB)) 
                #images.append(cv2.cvtColor(cv2.imread(right_current_path),cv2.COLOR_BGR2RGB))
                images.append((mpimg.imread(left_current_path))) 
                images.append((mpimg.imread(right_current_path)))
                measurements.append(steering_left)
                measurements.append(steering_right)
                
                ##Augmenting training data
                aug_images, aug_measurements = augment_image(images,measurements)
                
                ## Reducing low steering angle data 
                aug_images, aug_measurements, rem_list_rev = remove_low_steering_angle_data(aug_images, aug_measurements)

            #convert to numpy arrays
            dataX = np.array(aug_images)
            dataY = np.array(aug_measurements)
            
            yield shuffle(dataX, dataY)



# In[4]:

##Read input from csv file generated by simulator
with open(DATA_PATH) as csv_file:
    reader = csv.reader(csv_file)   
    for line in reader:
        lines.append(line)
        
# skip header row  
input_lines = lines[1:]

## Split training and validation datasets
training_count = int(0.8 * len(input_lines))
# training_data = input_lines[:training_count]
# validation_data = input_lines[training_count:]

training_data, validation_data = train_test_split(input_lines, train_size= training_count, random_state=42)
print("Length of original training/validation data:", len(training_data), len(validation_data))


# In[5]:

##Create model
model = nvidia_model()

# File to save the learned model
filepath="model_aws.h5"
# Callbacks for saving the model when val loss decreases
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#Early termination if val loss is not reducing uptil next 20 epochs
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

callbacks_list = [checkpoint, early_stopping]

training_datagen = image_data_generator(training_data, BATCH_SIZE)
validation_datagen = image_data_generator(validation_data, BATCH_SIZE)
samples_per_epoch = int(len(training_data) )
nb_val_samples = len(validation_data)

#Training the model using keras fit 
history_values = model.fit_generator(training_datagen, samples_per_epoch = samples_per_epoch, validation_data = validation_datagen, nb_epoch = EPOCHS, nb_val_samples=nb_val_samples, callbacks=callbacks_list, verbose=2)


# In[ ]:

# ### print the keys contained in the history object
# print(history_values.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_values.history['loss'])
# plt.plot(history_values.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()


# In[ ]:

#whos

