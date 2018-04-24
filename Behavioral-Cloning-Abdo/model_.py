
# coding: utf-8

# In[29]:


import csv
import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout


# In[3]:


# Lines in the csv file
lines = []

# Reading the csv file and save each line in lines array
with open ('TrainingAllV2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# Each line contain: Center image, left image, right image, steering value, throttle, brake, speed
# and the tag of each one is found in lines[0], so the data starts from lines[1]
# print("First row: ", lines[0])
# print("Second row (data): ", lines[1])
# print()
# print("Lines Number: ", len(lines)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            )


# In[4]:


'''
# Save the images with its meassurement in arrays

# Basic Datapath
datapath = 'data/'
# correction for the left and the right images of the sample data
steering_correction = 0.2
# Images array for the input data
images = []
# Measurements will be the labels for the Images
measurements = []

for line in lines[1:]:
    measurement = float(line[3])
    image = cv2.imread(datapath + line[0])
    
'''
f = 'TrainingAllV2'

# In[41]:


#Save the images with its meassurement in arrays

#correction for the left and the right images of the sample data
steering_correction = 0.2
#Images array for the input data
images = []
#Measurements will be the labels for the Images
measurements = []
counter = 0
for line in lines[1:]:
    counter = counter +1
    print(counter)
    # Normal Images
    center_path = line[0]
    left_path = line[1]
    right_path = line[2]
    cent_measurement = float(line[3])
    speed = float(line[6])/100.0;



    center_filename = (center_path.split('/')[-1]).split('\\')[-1]
    left_filename = (left_path.split('/')[-1]).split('\\')[-1]
    right_filename = (right_path.split('/')[-1]).split('\\')[-1]

    center_current_path = f + '/IMG/' + center_filename
    left_current_path = f + '/IMG/' + left_filename
    right_current_path = f + '/IMG/' + right_filename
    # print(center_current_path)
    center_image = cv2.imread(center_current_path)[..., ::-1]
    left_image = cv2.imread(left_current_path)[..., ::-1]
    right_image = cv2.imread(right_current_path)[..., ::-1]
    # cent_images_path = 'data/' + cent_image_name
    # cent_image = cv2.imread(cent_images_path)
    #
    # left_images_path = 'data/' + left_image_name
    # left_image = cv2.imread(left_images_path)
    left_measurement = cent_measurement + steering_correction
    
    # right_images_path = 'data/' + right_image_name
    # right_image = cv2.imread(right_images_path)
    right_measurement = cent_measurement - steering_correction
    
    #Flipped Images
    #cent_image_flipped = cv2.flip(cent_image,1)
    #cent_measurement_flipped = -cent_measurement
    
    #left_image_flipped = cv2.flip(left_image,1)
    #left_measurement_flipped = -left_measurement
    
    #right_image_flipped = cv2.flip(right_image,1)
    #right_measurement_flipped = -right_measurement

    center_image = cv2.resize(center_image[40:140, :], (64, 64))
    images.append(center_image)
    #images.append(cent_image_flipped)
    left_image = cv2.resize(left_image[40:140, :], (64, 64))
    images.append(left_image)
    #images.append(left_image_flipped)
    right_image = cv2.resize(right_image[40:140, :], (64, 64))
    images.append(right_image)
    #images.append(right_image_flipped)
    # print(center_image)
    measurements.append(tuple((cent_measurement, speed)))
    measurements.append(tuple((left_measurement, speed)))
    measurements.append(tuple((right_measurement, speed)))
    # measurements.append(cent_measurement)
    #measurements.append(cent_measurement_flipped)
    # measurements.append(left_measurement)
    #measurements.append(left_measurement_flipped)
    # measurements.append(right_measurement)
    #measurements.append(right_measurement_flipped)
    
#Expected to be (Lines Number-1) * 3 = 8036 * 6 = 24108
print("Images Number: ", len(images))
print("Measurement Number: ", len(measurements))
# plot samples from the Images
# plt.imshow(images[0])
# plt.show()

# In[6]:


# Convert images and measurements into numpy arrays for keras

# Covert to numpy arrays
images = np.array(images)
measurements = np.array(measurements)


# In[7]:


# Plot samples from the Images and Flip the images to increase data to achieve data augmentation

# plot samples from the Images
# plt.imshow(images[0])
# plt.show()

'''
# Flip images to increase data to achieve data augmentation 
images_flipped = np.array([])
measurements_flipped = np.array([])

for im_idx, im in enumerate(images):
    im_flipped = np.flip(im,1)
    meas_flipped = -measurements[im_idx]
    
    images_flipped = np.append(images_flipped, im_flipped)
    measurements_flipped = np.append(measurements_flipped, meas_flipped)


plt.imshow(images_flipped[1])
plt.show()
'''


# In[21]:


import tensorflow as tf

def Normalize_Centering(image):
    resized = image/255.0 - 0.5
    return resized


# In[32]:


# Normalization, Mean Centering, and Cropping

def Data_PreProcessing():
    Max_pixel = 255.0
    Centering = 0.5
    
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, name="image_normalization", input_shape=(64, 64, 3)))
    model.add(Cropping2D(cropping=((50,20),(0,0))))
    return model

# In[37]:
# from keras.layers.advanced_activations import RELU

# We will depend on Nvidia Model 

def Nvidia(input_shape=(160, 320, 3)):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, name="image_normalization", input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, name="convolution_1", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, name="convolution_2", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, name="convolution_3", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, name="convolution_4", border_mode="valid", init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, name="convolution_5", border_mode="valid", init='he_normal'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1000, name="hidden1", init='he_normal'))
    model.add(Dropout(p=0.5))
    model.add(Activation('relu'))
    model.add(Dense(500, name="hidden2", init='he_normal'))
    model.add(Dropout(p=0.5))
    model.add(Activation('relu'))
    model.add(Dense(100, name="hidden3", init='he_normal'))
    model.add(Dropout(p=0.2))
    model.add(Activation('relu'))
    model.add(Dense(10, name="hidden4", init='he_normal'))
    model.add(Dense(2, name="steering_angle_vel", activation="linear"))



    return model






def Nvidia_model():
    # Normalizing, Mean Cenetring, Cropping
    model = Data_PreProcessing()
    # Nvidia Architecture
    model.add(Convolution2D(nb_filter=24,nb_row=5,nb_col=5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(nb_filter=36,nb_row=5,nb_col=5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(nb_filter=48,nb_row=5,nb_col=5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(nb_filter=64,nb_row=3,nb_col=3,activation='relu'))
    model.add(Convolution2D(nb_filter=64,nb_row=3,nb_col=3,activation='relu'))
    model.add(Flatten())
    # Adding Dropout layer
    # model.add(Dropout(p=0.2))
    # model.add(Activation('relu'))
    # model.add(Dense(1000))
    # Adding Dropout layer
    # model.add(Dropout(p=0.5))
    # model.add(Activation('relu'))
    # model.add(Dense(500))
    
    # Adding Dropout layer
    # model.add(Dropout(p=0.5))
    # model.add(Activation('relu'))
    # model.add(Dense(100))
    
    # Adding Dropout layer
    # model.add(Dropout(p=0.2))
    # model.add(Activation('relu'))
    # model.add(Dense(10))
    model.add(Dense(1)) # Steering angle action
    
    return model


# In[38]:


# Training data

X_train = images
y_train = measurements


# In[40]:


# Run the model

Model = Nvidia(input_shape=(64, 64, 3))

# Mean square error and adam optimizer
Model.compile(loss='mse', optimizer='adam')

Model.summary()
# 15% validation and apply shuffling
epochs = 10
batch_size = 512
Model.fit(x=images, y=measurements, nb_epoch=epochs, batch_size=batch_size,  validation_split=0.2, shuffle=True)

# Save the model for testing it and give it as a paramter when running drive.py
Model.save('model_trial.h5')




