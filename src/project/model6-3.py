import cv2
import os
import pickle
import numpy as np
from tensorflow import keras
import tensorflow as tf

from utils import *
from data_gen import DataGenerator, PredictionDataGenerator

NAME = 'model6-3'
PADDING = 'padding3'


def create_model(input_shape=(48,48,3), optimizer=tf.train.AdamOptimizer, loss='binary_crossentropy', metrics=['accuracy']):
    model = keras.Sequential([
        keras.layers.Conv2D(20, (4,4), activation=tf.nn.relu, input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # keras.layers.Dropout(0.25),
        keras.layers.Conv2D(40, (3,3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # keras.layers.Dropout(0.25),
        keras.layers.Conv2D(60, (3,3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # keras.layers.Dropout(0.25),
        keras.layers.Conv2D(80, (2,2), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
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

#   model = Sequential()
# # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# # this applies 32 convolution filters of size 3x3 each.
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

md('checkpoints/{}'.format(NAME))
md('saved_models')
md('histories')

# Parameters
params = {'dim': (48,48),
          'batch_size': 200,
          'n_channels': 3,
          'shuffle': True,
          'datapath': f'/vagrant/imgs/training_data2/{PADDING}'}

# Datasets
partition = get_pickled_partition('all', PADDING)


# Generators
data_generators = {
    'training_generator': DataGenerator(partition['train'], **params),
    'validation_generator': DataGenerator(partition['validation'], **params),
    'test_generator': DataGenerator(partition['test'], **params),
    'predition_generator': PredictionDataGenerator(partition['test'], dim=(48,48))
}

model = create_model()
print("EVALUATING")
history, result, predictions = evaluate_model(model, data_generators, 'checkpoints/model6-3/cp-{epoch:04d}.ckpt', period=1, epochs=40, patience=5)

print("SAVING HISTORY")
with open('histories/{}.pkl'.format(NAME), 'wb') as f:
    pickle.dump(history.history, f)
print("SAVING RESULTS")
with open('evaluation_results/{}.pkl'.format(NAME), 'wb') as f:
    pickle.dump(result, f)
print("SAVING PREDICTIONS")
with open('predictions/{}.pkl'.format(NAME), 'wb') as f:
    pickle.dump(predictions, f)
print("SAVING MODEL")
model.save('saved_models/{}.h5'.format(NAME))

# with open('histories/model2.pkl', 'rb') as f:
#     history = pickle.load(f)
