#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import glob
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[3]:


#Function to load image is it's within the IMG folder
def load_img(filename):
    filename = filename.strip()
    if filename.split('/')[0] == 'IMG':
        filename = '/opt/carnd_p3/data/{}'.format(filename)
    return plt.imread(filename)


# In[ ]:


# Function to build network
# I've based this on the Nvidia network mentioned in the theory sections. This 
def net():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0)-0.5))
    model.add(Conv2d(24, (5,5), strides= (2,2), activation = 'relu'))
    model.add(Conv2d(36, (5,5), strides= (2,2), activation = 'relu'))
    model.add(Conv2d(48, (3,3), strides= (2,2), activation = 'relu'))
    model.add(Conv2d(64, (3,3), strides= (2,2), activation = 'relu'))
    model.add(Conv2d(64, (5,5), strides= (2,2), activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    return model

def generator(image_path, angles, batch_size);
    images = np.empty([batch_size, img_height, img_width, img_channel])
    measurements = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_path.shape[0]):
            center, left, right = image_path[index]
            image = load_image(center)
            measurement = float(angles[index])

            image, measurement = augment(image, measurement)

            images[i] = image
            measurements[i] = measurement

            i += 1
            if i == batch_size:
                break
        yield images, measurements

def main():
    """Entry point for training the model"""

    # Hyperparameters
    test_size = 0.20
    batch_size = 256
    epochs = 5
    verbose = 1
    additional_training_data = True

    print('Loading data...')
    X_train, X_val, y_train, y_val = load_data(test_size, additional_training_data=additional_training_data)

    print('Building model...')
    model = build_model()

    checkpoint = ModelCheckpoint('model_checkpoints/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    print('Compiling model...')
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

    print('Training model...')
    history_object = model.fit_generator(batch_generator(X_train, y_train, batch_size, correction),
                                         steps_per_epoch=len(X_train)/batch_size,
                                         validation_data=batch_generator(X_val, y_val, batch_size, correction),
                                         validation_steps=len(X_val)/batch_size,
                                         callbacks=[checkpoint],
                                         epochs=epochs,
                                         verbose=verbose)

    print('Saving model...')
    model.save('model.h5')

    print('Model saved, training complete!')

    # summarize history for loss
    plt.subplot(111)
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('history.png', bbox_inches='tight')

if __name__ == '__main__':
    main()