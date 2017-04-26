##Pre-processing steps
import tensorflow as tf
import numpy as np
import cv2

def image_crop(img):
    #crop the top 50 pixels
    return img[50:,:]

def image_resize(img):
    return tf.image.resize_images(img, (66,200))

# def remove_low_steering_angle_data(X_train, y_train):
#     rem_tuple = np.where(np.logical_and(y_train>=-0.05,y_train<=0.05))
#     rem_list = rem_tuple[0]   
#     print(rem_list)
#     print("length of remlist:", rem_list.shape)
#     print("old shape of X_train, y_train", X_train.shape, y_train.shape)
#     y_train_new = np.delete(y_train, rem_list[])
#     for k in range(0,len(X_train)):
#         #if k in rem_list:
#         if(np.any(rem_list) == k):
#             X_train = (np.delete(X_train, k))
#     print("new shape of X_train, y_train", X_train.shape, y_train_new.shape)
#     return X_train, y_train_new
    
    
def augment_image(images,measurements):
    augmented_images,augmented_measurements = [],[]
    print("1",len(images), len(measurements))
    i = 0
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        #flip the images horizontally
        augmented_images.append(cv2.flip(image,1))
        #invert the steering angles
        augmented_measurements.append(-measurement)
    print("loop i: ", i+1, len(augmented_images), len(augmented_measurements))
    return augmented_images,augmented_measurements
    