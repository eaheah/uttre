import cv2
import os
import pickle
import numpy as np
from tensorflow import keras
import tensorflow as tf

from utils import *
from data_gen import DataGenerator, PredictionDataGenerator

NAME = 'model7-3'
PADDING = 'padding3'

def create_model(input_shape=(48,48,3), optimizer=keras.optimizers.Adam, loss='binary_crossentropy', metrics=['accuracy']):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (4,4), activation=tf.nn.relu, input_shape=input_shape),
        keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
        keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(40, activation=tf.nn.sigmoid)
      ])
    model.compile(optimizer=optimizer(), 
                  loss=loss,
                  metrics=metrics)    
    return model

def compile_model(model, optimizer=keras.optimizers.Adam, loss='binary_crossentropy', metrics=['accuracy']):
    model.compile(optimizer=optimizer(), 
                  loss=loss,
                  metrics=metrics)    
    return model    

def _fit_model(model, data_generators, checkpoint_path, patience=20, period=5, workers=8, epochs=100, verbose=1):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=verbose, save_weights_only=True,
        period=period)
    
    history = model.fit_generator(generator=data_generators['training_generator'],
                        validation_data=data_generators['validation_generator'],
                        use_multiprocessing=True,
                        workers=workers,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=[early_stop, cp_callback])

    return history

md('checkpoints/{}'.format(NAME))
md('saved_models')
md('histories')

# Parameters
params = {'dim': (48,48),
          'batch_size': 10,
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
print("FITTING")
history = _fit_model(model, data_generators, 'checkpoints/model7-3/cp-{epoch:04d}.ckpt', period=1, epochs=40, patience=10)

print("SAVING HISTORY")
with open('histories/{}.pkl'.format(NAME), 'wb') as f:
    pickle.dump(history.history, f)
print("SAVING MODEL")
model.save('saved_models/{}.h5'.format(NAME))

print("EVALUATING")
result = model.evaluate_generator(generator=data_generators['test_generator'], verbose=1)
print("PREDICTING")
predictions = model.predict_generator(generator=data_generators['predition_generator'], verbose=1)

# with open('histories/model2.pkl', 'rb') as f:
#     history = pickle.load(f)
