#Import packages
import cv2
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout

# Setting hyperparameters
test_size = 0.10
batch_size = 64
epochs = 9

from sklearn.model_selection import train_test_split

def load_data(test_size):
    """ 
    Load and split data into test & training sets
    """
    header = None
    names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv('drivedata/driving_log.csv',
                          header=header, names=names)

    data_df = data_df[(data_df.center != 'center') |
                      (data_df.left != 'left') |
                      (data_df.right != 'right')]

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    return train_test_split(X, y, test_size=test_size)

def load_image(image_file):
    """ 
    Find images in img folder
    """
    image_file = image_file.strip()
    if image_file.split('/')[0] == 'IMG':
        image_file = 'drivedata/{}'.format(image_file)
    return mpimg.imread(image_file)

def augment_img(img, angles):
    """
    Flip images if random number<0.5.
    If new random numer is <0.5 as well, change brightness
    """
    if np.random.rand() < 0.5:
        img = np.fliplr(img)
        angles = -angles
    if np.random.rand() < 0.5:
        delta_pct = random.uniform(0.4, 1.2)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * delta_pct
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img, angles


def nvidia_net():
    """
    Build keras model using NVIDIA autonomous vehicle architecture
    """
    model = Sequential()
    model.add(Lambda(lambda x:(x/255.0)-0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def generator(img_path, steering_angles, batch_size):
    """
    Returns arrays with images and the corresponding steering angles for training
    """
    images = np.empty([batch_size, 160, 320, 3])
    angles = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(img_path.shape[0]):
            center, left, right = img_path[index]
            img = load_image(center)
            angle = float(steering_angles[index])

            img, angle = augment_img(img, angle)

            images[i] = img
            angles[i] = angle

            i += 1
            if i == batch_size:
                break
        yield images, angles


def main():
    
    print('Loading data.')
    
    X_train, X_val, y_train, y_val = load_data(test_size)

    print('Building model')
    
    model = nvidia_net()

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    print('Compiling')
    
    model.compile(loss='mse', optimizer='adam')

    print('Training')
    
    history_object = model.fit_generator(generator(X_train, y_train, batch_size),
                                         steps_per_epoch=len(X_train)/batch_size,
                                         validation_data=generator(X_val, y_val, batch_size),
                                         validation_steps=len(X_val)/batch_size,
                                         callbacks=[checkpoint],
                                         epochs=epochs,
                                         verbose=1)

    model.save('model.h5')
    
    print('Model trained and saved')

    # Print loss
    print('Loss: '+str(history_object.history['loss']))
    print('Validation Loss: '+str(history_object.history['val_loss']))

    # Plot loss graph
    plt.subplot(111)
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model MSE loss')
    plt.ylabel('MSE loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('history.png', bbox_inches='tight')

if __name__ == '__main__':
    main()