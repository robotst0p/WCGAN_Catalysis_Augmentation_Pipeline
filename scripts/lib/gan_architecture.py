'''

GAN model and util functions

This code follows "Advanced Deep Learning with Keras" util functions for WGAN at
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''


from tensorflow.keras.layers import Activation, Dense, Input, Lambda, Softmax
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as tl

import numpy as np
import math
import os

from tensorflow.keras.models import load_model


def generator(inputs,labels,
              n_of_suvr,
              activation='sigmoid'):
    """
    Generator Model

    Stack of MLP to generate suvr uptake data.
    Output activation is softmax.

    # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        n_of_suvr (int): Target number of suvr uptake data samples to generate
        activation (string): Name of output activation layer

    # Returns
        Model: Generator Model
    """
    
    ##################################### cGAN #################################
    
    x = concatenate([inputs, labels], axis=1)
    
    x = Dense(512)(x) # scale down 1/2
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(2*n_of_suvr)(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((2,n_of_suvr))(x)

    if activation is not None:
        if activation == 'softmax':
            x = Softmax(axis=1)(x)
        else:
            x = Activation(activation)(x)

    x = Lambda(lambda x: x[:,1])(x)

    return  Model([inputs, labels], x, name='generator')

#create generator model for gan in more tensorthonic way 
def make_generator_model(inputs, labels, n_of_suvr, activation = 'sigmoid'):
    model = tf.keras.Sequential()
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(2*n_of_suvr))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape(2,n_of_suvr))
    
    model.add(layers.Softmax(axis = 1))
    
    #final lambda layer
    model.add(layers.Lambda(lambda x: x[:,1]))
    
    return model
    
#create the discriminator model in more tensorthonic way
def make_discriminator_model(inputs, labels, n_of_suvr, activation = 'sigmoid'):
    model = tf.keras.Sequential()
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(0.2))
    
    #add final output layer and then activation function
    model.add(layers.Dense(1))
    
    model.add(layers.Activation('sigmoid'))
    
    return model


def discriminator(inputs,labels,
                  n_of_suvr,
                  activation='sigmoid'):
    
    x = inputs
    
    y = Dense(4)(labels) # 8 is the latent size
    #y = Reshape((100, 1))(y)
    
    x = concatenate([x, y])
    
    x = Dense(512)(x)  # scale down
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)

    outputs = Dense(1)(x)
    
    if activation is not None:
        print(activation)
        outputs = Activation(activation)(outputs)
    
    return  Model([inputs, labels], outputs, name='discriminator')
    

def test_generator(generator, class_label=None, num_samples = 100):
    n_sample = num_samples
    noise_input = np.random.uniform(-1.0, 1.0, size=[n_sample, 4]) # 4 is latent size
    
    if class_label is None:
        num_labels = 2
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, n_sample)]
    else:
        noise_class = np.zeros((n_sample, 2))
        noise_class[:,class_label] = 1
    
    #set verbose to 0 if you dont want predict call to print duration and time of generation
    suvr_data = generator.predict([noise_input, noise_class], verbose = 0)
    noise_class2 = np.argmax(noise_class, axis=1)
    return suvr_data, noise_class2
