import cv2
import os
import pickle
import numpy as np
from tensorflow import keras
import tensorflow as tf

from utils import *
from data_gen import DataGenerator, PredictionDataGenerator

NAME = 'model2'

def create_model(input_shape=(95,95,3), optimizer=tf.train.AdamOptimizer, loss='binary_crossentropy', metrics=['accuracy']):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(64, activation=tf.nn.relu),
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

md('checkpoints/{}'.format(NAME))
md('saved_models')
md('histories')

# Parameters
params = {'dim': (95,95),
          'batch_size': 100,
          'n_channels': 3,
          'shuffle': True}

# Datasets
partition = create_partition()


# Generators
data_generators = {
    'training_generator': DataGenerator(partition['train'], **params),
    'validation_generator': DataGenerator(partition['validation'], **params),
    'test_generator': DataGenerator(partition['test'], **params),
    'predition_generator': PredictionDataGenerator(partition['test'])
}

model = create_model()
history, result, predictions = evaluate_model(model, data_generators, 'checkpoints/model2/cp-{epoch:04d}.ckpt', period=1, epochs=40, patience=5)

model.save('saved_models/{}.h5'.format(NAME))
with open('histories/{}.pkl'.format(NAME), 'wb') as f:
    pickle.dump(history.history, f)
# with open('histories/model2.pkl', 'rb') as f:
#     history = pickle.load(f)
