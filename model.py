""" Work with generators.
Can compares more than one models.
Adds flipped images for images that has an absolute value of steering angle
 that is greater than rthr.
"""

import os

import csv
import numpy as np
from sklearn.utils import shuffle

import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers import Cropping2D
from keras.layers import ELU

import json
from datetime import datetime


# folder for recording the model /outputs/model/
# Records at each EPOCH
outparent = "model"
# Cropping shape for Keras Cropping layer
cropping_tuple = ((54, 0), (0, 0))
max_epoc = 50
val_sz = 0.2

# Set true to load weights for transfer learning
will_load = False
# model will be loaded from ./outputs/[outparent]/[load_subfolder]/model.json
load_subfolder = 'bc16/11'
# EPOCH of the loaded model
load_from = 0

# Add default images and flipped images
folders_reverse = []
# Add only default images
folders = []

# Images that has a steering angle greater than rthr will be added twice
# One for default, and one for flipped.
# The value is in degrees, as displayed in the simulator.
# It will be divided by 25.05
# before comparing with the y (steering_angle) values.
rthr = 0.0


folders_reverse = ['./data/data/'
                   ,'./debug/5/'
                   #,'./debug/6/'
                   ,'./debug/7/'
                   #,'./debug/8/'
                   #,'./debug/9/'
                   ,'./debug/10/'
                   ,'./debug/11/'
                   #,'./debug/12/'
                   #,'./recordc/1/'
                   #,'./recordn/1/'
                   #,'./recordn/2/'
                   ,'./records/1/'
                   ,'./records/2/'
                   #,'./records/3/'
                   ,'./records/4/'
                   ,'./records/5/'
                   ,'./records/6/'
                   ,'./records/7/'
                   ,'./records/8/'
                   ,'./records/9/'
                   ]


def get_gens(fold_rev, fold, val_sz=0.2, rthr=0.0):
    """
    Returns
    trn_gen: train data generator
    val_gen: validation data generator
    """
    samples = get_samples(fold_rev, fold, rthr=rthr)
    trn, val = train_test_split(samples,
                                test_size=val_sz)

    n_trn, n_val = len(trn), len(val)

    trn_gen = generator(trn, batch_size=32)
    val_gen = generator(val, batch_size=32)

    return n_trn, n_val, trn_gen, val_gen


def get_samples(fold_rev, fold, rthr=0.0):
    """
    # Returns samples from given folders
    # For fold_rev adds default images and flipped images
    # For fold adds only default images
    """
    samples = []
    for folder in fold_rev:
        samples += get_folder(folder, reverse=True, rthr=rthr)
    for folder in fold:
        samples += get_folder(folder, reverse=False)
    return samples


def generator(samples, batch_size=32):
    """
    Generator to load the images for samples in batches
    """
    num_samples = len(samples)
    while 1:  # Used as a reference pointer so code always loops back around
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                # center_image = cv2.imread(name)
                center_image = mpimg.imread(name)
                if batch_sample[-1]:
                    center_image = np.fliplr(center_image)
                center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            # X_train = X_train[:, 54:, :, :]
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def get_folder(folder='./data/data/', reverse=False, rthr=0.0):
    """
    Loads data from given folder.
    Required to have driving_log.csv file and IMG subfolder in the folder
    For images that has a steering angle absolute value greater than rthr
     flipped images are also added.
    Prints:
     # of Images in folder (n)
     # of samples added (t)
     # of flipped images added (f)
     # of default images added (r)
    """
    reverse_thr = float(rthr) / 25.05
    samples = []
    n = 0
    t = 0
    r = 0
    f = 0
    with open(os.path.join(folder, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0] == 'center':
                continue
            lline = []
            if folder[2:] not in line[0]:
                line[0] = folder + line[0].strip()
            lline.append(line[0])
            y = float(line[3])
            lline.append(y)
            lline.append(False)
            samples.append(lline)
            t += 1
            r += 1
            if reverse:
                yabs = np.abs(y)
                if yabs > reverse_thr:
                    rline = []
                    rline.append(lline[0])
                    rline.append(-y)
                    rline.append(True)
                    samples.append(rline)
                    t += 1
                    f += 1
            n += 1
    print(folder, 'Images:', n, ',Samples:', t, ',Flipped:', f, 'Default:', r)
    return samples


def get_modeln():
    """
    Creates a model based on NVIDIA model.
    Cropps image to have a size of 106 x 320
    Normalizes as in comma.ai model
    5 Convolution Layers with relu activation
    3 Fully Connected Layers
    6 Dropout layers
    adam optimizer
    mse loss function
    """
    ch, row, col = 3, 106, 320  # camera format

    model = Sequential()
    model.add(Cropping2D(cropping=cropping_tuple,
                         input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(24, 11, 11, subsample=(3, 3), border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def get_models():
    """
    Faster version of modeln1
    Creates a model based on NVIDIA model.
    Cropps image to have a size of 106 x 320
    Normalizes as in comma.ai model
    4 Convolution Layers with relu activation
    3 Fully Connected Layers
    5 Dropout layers
    adam optimizer
    mse loss function
    """
    ch, row, col = 3, 106, 320  # camera format

    model = Sequential()
    model.add(Cropping2D(cropping=cropping_tuple,
                         input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 10, 10, subsample=(4, 4), border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 7, 7, subsample=(3, 3), border_mode="valid"))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(1, 1), border_mode="valid"))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def get_modelc():
    """
    Creates a model based on comma.ai model.
    No cropping
    Normalizes as in comma.ai model
    3 Convolution Layers with ELU activation
    1 Fully Connected Layers
    adam optimizer
    mse loss function
    """
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def load_model():
    """
    Loads a previously saved model
    Loads from ./outputs/[outparent]/[load_subfolder]/model.json
    Returns the model if it is loaded, and None otherwise.
    """
    try:
        load_folder = './outputs/' + outparent + '/' + load_subfolder + '/'
        load_model = load_folder + 'model.json'
        load_weight = load_folder + 'model.h5'
        with open(load_model, 'r') as jfile:
            model = model_from_json(json.load(jfile))

        model.compile("adam", "mse")

        model.load_weights(load_weight)

        print("Model loaded")
        return model

    except:
        print("Model not loaded")
        return None


n_trn, n_val, trn_gen, val_gen = None, None, None, None


def fit_model(fmodel, current_epoch=0):
    """
    Runs Keras fit_generator for one epoch.
    current_epoch indicates the current epoch number.
    """
    epoc = current_epoch
    if will_load:
        epoc += load_from

    history = fmodel.fit_generator(trn_gen,
                                   samples_per_epoch=n_trn,
                                   validation_data=val_gen,
                                   nb_val_samples=n_val,
                                   nb_epoch=1 + epoc,
                                   initial_epoch=epoc)
    return history


n_trn, n_val, trn_gen, val_gen = get_gens(folders_reverse,
                                          folders,
                                          val_sz=val_sz,
                                          rthr=rthr
                                          )

t0 = datetime.now()
model = None

M = []
mnames = []

# load for only one model
if will_load:
    print('loading models')
    model = load_model()
    M.append(model)
    mnames = ['loaded']

if model is None:
    print('creating models')
    M.append(get_modeln())
    print('modeln1 created')
    M.append(get_models())
    print('models1 created')
    M.append(get_modelc())
    print('modelc1 created')

    mnames = ['n1', 's1', 'c1']

n_model = len(mnames)

# To keep train loss and validation loss data for each model
min_loss = [999999] * n_model
min_t_loss = [999999] * n_model
min_loss_ep = [-1] * n_model
min_t_loss_ep = [-1] * n_model

print('n_trn:', n_trn, ', n_val:', n_val, ', n_model:', n_model)

# Creates required folders
outparentfold = "./outputs/" + outparent
if not os.path.exists('./outputs'):
    os.makedirs('./outputs')
if not os.path.exists(outparentfold):
    os.makedirs(outparentfold)

# To record train loss and validation loss data for each model and epoch
# recreates result.csv file
fresult = open(outparentfold + '/result.csv', 'w')
fresult.close()
fresult = open(outparentfold + '/result.csv', 'a')


# For each epoch and for each model saves the model and weights to folder
# ./outputs/[outparent]/[modelname]_[epoch_num]/

for i in range(max_epoc):
    trn_losses = [0.0] * n_model
    val_losses = [0.0] * n_model

    for j in range(len(M)):
        model = M[j]

        history = fit_model(model, i)
        val_loss = history.history['val_loss'][-1]
        trn_loss = history.history['loss'][-1]
        val_losses[j] = val_loss
        trn_losses[j] = trn_loss

        if trn_loss < min_t_loss[j]:
            min_t_loss[j] = trn_loss
            min_t_loss_ep[j] = i

        if val_loss < min_loss[j]:
            min_loss[j] = val_loss
            min_loss_ep[j] = i

        if will_load:
            outfold = outparentfold + "/" + str(load_from) + \
                '_' + mnames[j] + '_' + str(i)
        else:
            outfold = outparentfold + "/" + mnames[j] + '_' + str(i)
        if not os.path.exists(outfold):
            os.makedirs(outfold)

        model.save_weights(outfold + "/model.h5", True)
        with open(outfold + '/model.json', 'w') as outfile:
            json.dump(model.to_json(), outfile)

    for j in range(len(M)):
        print(i, mnames[j], trn_losses[j], val_losses[j])
        fresult.write('%s,%s,%d,%f,%f\n' %
                      (outparent, mnames[j], i, trn_losses[j], val_losses[j]))
    print(i, 'time:', datetime.now() - t0)

for j in range(len(M)):
    print(mnames[j], 'trn:', min_t_loss[j], 'val:', min_loss[j],
          'epochs:', min_t_loss_ep[j], min_loss_ep[j])


fresult.close()
