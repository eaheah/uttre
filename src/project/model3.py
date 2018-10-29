import cv2
import os
import pickle
import numpy as np
from tensorflow import keras
import tensorflow as tf

from utils import *
from data_gen import DataGenerator, PredictionDataGenerator

NAME = 'model3'

def create_model(input_shape=(30,30,3), optimizer=tf.train.AdamOptimizer, loss='binary_crossentropy', metrics=['accuracy']):
    model = keras.Sequential([
        keras.layers.Conv2D(20, (4,4), activation=tf.nn.relu, input_shape=input_shape),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(40, (3,3), activation=tf.nn.relu),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(60, (3,3), activation=tf.nn.relu),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(80, (2,2), activation=tf.nn.relu),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(40, activation=tf.nn.sigmoid)
      ])
    model.compile(optimizer=optimizer(), 
                  loss=loss,
                  metrics=metrics)    
    return model

def compile_model(model, optimizer=tf.train.AdamOptimizer, loss='binary_crossentropy', metrics=['accuracy']):
    model.compile(optimizer=optimizer(), 
                  loss=loss,
                  metrics=metrics)    
    return model    

if __name__ == '__main__':

    md('checkpoints/{}'.format(NAME))
    md('saved_models')
    md('histories')

    # Parameters
    params = {'dim': (30,30),
              'batch_size': 200,
              'n_channels': 3,
              'shuffle': True}

    # Datasets
    partition = create_partition(10000)


    # Generators
    data_generators = {
        'training_generator': DataGenerator(partition['train'], **params),
        'validation_generator': DataGenerator(partition['validation'], **params),
        'test_generator': DataGenerator(partition['test'], **params),
        'predition_generator': PredictionDataGenerator(partition['test'], dim=(30,30))
    }

    model = create_model()
    history, result, predictions = evaluate_model(model, data_generators, 'checkpoints/model3/cp-{epoch:04d}.ckpt', period=1, epochs=40, patience=5)

    model.save('saved_models/{}.h5'.format(NAME))
    with open('histories/{}.pkl'.format(NAME), 'wb') as f:
        pickle.dump(history.history, f)
    # with open('histories/model2.pkl', 'rb') as f:
    #     history = pickle.load(f)
