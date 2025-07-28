'''

Trains WGAN on lung somatic mutation profiles using Tensorflow Keras

This code follows "Advanced Deep Learning with Keras" training of WGAN at
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

#from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from keras.callbacks import TensorBoard

import numpy as np
import argparse
import pandas as pd
import datetime


from lib import gan_architecture as gan
        
def write_log(writer, d_loss, a_loss, i):
    with writer.as_default():
        tf.summary.scalar('Discriminator_loss', d_loss,  step = i)
        tf.summary.scalar( 'Adversarial_loss', a_loss , step = i)
        #tf.summary.scalar('Discriminator_loss', d_loss, 'Adversarial_loss', a_loss )

        writer.flush() 
        
def train(models, data, params):
    """
    Train function for the Discriminator and Adversarial Networks

    It first trains Discriminator with real and fake suvr uptake data for the cingulate region of the brain
    Discriminator weights are clipped as a requirement of Lipschitz constraint.
     is trained next (via Adversarial) with fake suvr uptake data
    pretending to be real.
    Generate sample uptake data per save_interval

    # Arguments
        models (list): Generator, Discriminator, Adversarial models
        x_train (tensor): Train suvr data
        params (list) : Networks parameters

    """
    
    generator, discriminator, adversarial = models
    x_train, y_train = data
    
    writer = tf.summary.create_file_writer("C:/Users/meyer/Desktop/SUVr_Analysis/logs/log_cingulate_batch3_latent4_lr-5e-5")

    # network parameters
    (batch_size, latent_size, n_critic, clip_value, train_steps, model_name) = params
    num_labels = 2
    
    # setting up a save interval to be every 500 steps
    save_interval = 500
    
    # number of elements in train dataset
    train_size = x_train.shape[0]
    
    # labels for real data
    real_labels = np.ones((batch_size, 1))
    
    
    for i in range(train_steps):
        # train discriminator n_critic times
        loss = 0
        acc = 0
        for _ in range(n_critic):
            # train the discriminator for 1 batch
            # 1 batch of real (label=1.0) and fake suvr data (label=-1.0)
            # randomly pick real suvr data from dataset
            rand_indexes = np.random.randint(0, train_size, size=batch_size)
            real_suvr = x_train[rand_indexes]
            
            real_labels = y_train[rand_indexes]
            
            # generate fake suvr data from noise using generator
            # generate noise using uniform distribution
            noise = np.random.uniform(-1.0,
                                      1.0,
                                      size=[batch_size, latent_size])
            
            # assign random one-hot labels
            fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
            
            # generate fake images conditioned on fake labels
            fake_suvr = generator.predict([noise, fake_labels])



            # train the discriminator network
            # real data label=1, fake data label=-1
            
            real_loss, real_acc = discriminator.train_on_batch([real_suvr, real_labels], np.ones([batch_size, 1]) )           
            
           
            fake_loss, fake_acc = discriminator.train_on_batch([fake_suvr,fake_labels ], -np.ones([batch_size, 1]))
            
            
            # accumulate average loss and accuracy
            loss += 0.5 * (real_loss + fake_loss)
            acc += 0.5 * (real_acc + fake_acc)

            # clip discriminator weights to satisfy Lipschitz constraint
            for layer in discriminator.layers:
                weights = layer.get_weights()
                weights = [np.clip(weight,
                                   -clip_value,
                                   clip_value) for weight in weights]
                layer.set_weights(weights)

        # average loss and accuracy per n_critic training iterations
        loss /= n_critic
        acc /= n_critic
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
        
        
        
        
        # train the adversarial network for 1 batch
        # 1 batch of fake images with label=1.0
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        
        # assign random one-hot labels
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]
        
        # train the adversarial network
        # log the loss and accuracy
        ad_loss, ad_acc = adversarial.train_on_batch([noise,fake_labels], np.ones([batch_size, 1]))
                
        log = "%s [adversarial loss: %f, acc: %f]" % (log, ad_loss, ad_acc)
        print(log)
        
        
        #### tensorboard
        #write_log(writer_a, writer_d, loss, ad_loss, i)
        print(i)
        
        #write_log is responsible for tensorboard graphing
        write_log(writer, loss, ad_loss, i)

        
        if (i + 1) % save_interval == 0:
            generator.save("weights/"+model_name+"_"+str(i)+".h5")

    # save the model after training the generator
    # the trained generator can be reloaded for future MNIST digit generation
    generator.save("weights/"+model_name + ".h5")


def wasserstein_loss(y_label, y_pred):
    """
    Implementation of a Wasserstein Loss with keras backend
    """
    return -K.mean(y_label * y_pred)


def build_and_train_models():
    raw_dataframe = pd.read_excel('C:/Users/meyer/Desktop/SUVr_Analysis/original_data/AUD_SUVR_wb_cingulate.xlsx') #,index_col = 0

    
    raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
    raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0
    
    y_train = raw_dataframe['CLASS']
    
    num_labels = np.amax(y_train) + 1
    y_train = to_categorical(y_train)

    
    
    x_train = raw_dataframe.drop(['CLASS', 'Subj_id'], axis = 1).to_numpy()


    model_name = "wgan_CingulateSUVR"
    # network parameters
    # the latent or z vector is 500-dim
    latent_size = 4 #500
    # hyper parameters
    n_critic = 5
    clip_value = 0.01
    batch_size = 27#64
    lr = 5e-5
    train_steps = 300000
    
    n_samples = x_train.shape[1]   
    input_shape = (n_samples, )   
    label_shape = (num_labels, )
    
    
    ################################################ build discriminator model
    
    inputs = Input(shape=input_shape, name='discriminator_input')
    labels = Input(shape=label_shape, name='class_labels')
    
    # WGAN uses linear activation
    discriminator = gan.discriminator(inputs, labels, n_samples, activation='linear')
    #optimizer = RMSprop(lr=lr)
    
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr)

    # WGAN discriminator uses wassertein loss
    discriminator.compile(loss=wasserstein_loss,
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()


    #################################################### build generator model
    
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = gan.generator(inputs, labels, n_samples, activation = 'softmax')
    generator.summary()


    ###################### build adversarial model = generator + discriminator
    
    # freeze the weights of discriminator during adversarial training
    discriminator.trainable = False
    
    op = discriminator([generator([inputs, labels]), labels])
    
    adversarial = Model([inputs,labels],
                        op,
                        name=model_name)
    
    adversarial.compile(loss=wasserstein_loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()


    ############################# train discriminator and adversarial networks
    models = (generator, discriminator, adversarial)
    
    data = (x_train, y_train)
    
    params = (batch_size,
              latent_size,
              n_critic,
              clip_value,
              train_steps,
              model_name)
    
    train(models, data, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        # np.save("generated_luad"+str(time.strftime("%m|%d|%y_%H%M%S")) +".npy",\
        #          gan.test_generator(generator));
        np.save("generated_cingulate_SUVR.npy", gan.test_generator(generator));    
    else:
        build_and_train_models()
