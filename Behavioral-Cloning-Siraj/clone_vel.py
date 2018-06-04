import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
import random
from matplotlib import pyplot as plt

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def flip(image, angle):
  new_image = cv2.flip(image,1)
  new_angle = angle*(-1)
  return new_image, new_angle





def data_generator(image_paths, angles, batch_size=32):
    x = np.zeros((batch_size, 160, 320, 3), dtype=np.uint8)
    y = np.zeros(batch_size)
    while True:
        data, angle = shuffle(image_paths, angles)
        for i in range(batch_size):
            choice = int(np.random.choice(len(data), 1))
            x[i] = cv2.imread(data[choice])
            y[i] = angle[choice]
            # Flip random images#
            flip_coin = random.randint(0, 1)
            if flip_coin == 1:
                x[i], y[i] = flip(x[i], y[i])
        yield x, y




images = []
images_paths = []

measurements = []


files = []

files.append('TrainingV3')


counter = 0
for f in files:
    lines = []
    with open(f+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines[1:]:
        center_path = line[0]
        left_path = line[1]
        right_path = line[2]

        center_filename = (center_path.split('/')[-1]).split('\\')[-1]
        left_filename = (left_path.split('/')[-1]).split('\\')[-1]
        right_filename = (right_path.split('/')[-1]).split('\\')[-1]

        center_current_path = f + '/IMG/' + center_filename
        left_current_path = f + '/IMG/' + left_filename
        right_current_path = f + '/IMG/' + right_filename

        center_image = cv2.imread(center_current_path)[...,::-1]
        left_image = cv2.imread(left_current_path)[...,::-1]
        right_image = cv2.imread(right_current_path)[...,::-1]

        measurement = float(line[3])
        speed = translate(float(line[6]), 0, 60, -1, 1)
        # print(speed)
        flip_coin = random.randint(0, 1)
        if flip_coin == 1:
            flipped_center_image, flipped_measurement = flip(center_image, measurement)
            flipped_center_image = cv2.resize(flipped_center_image[40:140, :], (64, 64))
            images.append(flipped_center_image)
            images_paths.append(center_current_path)
            # measurements.append(tuple((flipped_measurement, speed)))
            measurements.append(speed)

        center_image = cv2.resize(center_image[40:140, :], (64, 64))
        images.append(center_image)
        images_paths.append(center_current_path)
        # measurements.append(tuple((measurement, speed)))
        measurements.append(speed)

        left_image = cv2.resize(left_image[40:140, :], (64, 64))
        images.append(left_image)
        images_paths.append(left_current_path)
        # measurements.append(tuple((measurement+0.22, speed)))
        measurements.append(speed)

        right_image = cv2.resize(right_image[40:140, :], (64, 64))
        images.append(right_image)
        images_paths.append(right_current_path)
        # measurements.append(tuple((measurement-0.22, speed)))
        measurements.append(speed)
        counter += 1
        print(counter)

X_train = np.array(images)
print(X_train.shape)

y_train = np.array(measurements)
print(y_train.shape)
# print(y_train[0])

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout, Cropping2D, Activation
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU


def build_model(input_shape, keep_prob):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def Nvidia(input_shape=(160, 320, 3)):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, name="image_normalization", input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, name="convolution_1", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, name="convolution_2", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, name="convolution_3", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, name="convolution_4", border_mode="valid", init='he_normal'))
    model.add(ELU())
    # model.add(Convolution2D(64, 3, 3, name="convolution_5", border_mode="valid", init='he_normal'))
    # model.add(ELU())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, name="hidden1", init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(50, name="hidden2", init='he_normal'))
    model.add(ELU())
    model.add(Dense(10, name="hidden3", init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, name="vel", activation="linear"))

    return model

epochs_arr = [200, 300, 500]

for x in range(0, len(epochs_arr)):
    model = build_model(input_shape=(64, 64, 3), keep_prob=0.5)
    learning_rate = 0.001
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mse')
    model.summary()
    epochs = epochs_arr[x]
    batch_size = 512
    model.fit(x=X_train, y=y_train, nb_epoch=epochs, batch_size=batch_size,  validation_split=0.2, shuffle=True)
    print(model.predict(X_train))
    model.save('model_5_vel_' + str(learning_rate) + 'lr_' + str(epochs) + 'epoch.h5')



