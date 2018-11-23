from utils import *
from data_gen import DataGenerator, PredictionDataGenerator
from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np

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

params = {'dim': (48,48),
          'batch_size': 10,
          'n_channels': 3,
          'shuffle': True,
          'datapath': f'/vagrant/imgs/training_data2/{PADDING}'}


partition = get_pickled_partition('all', PADDING)

print(len(partition['test']))
prediction_data = []

for i in range(0, len(partition['test']), 200):
	prediction_data.append(partition['test'][i:i+200])
print(len(prediction_data))
# 1/0
data_generators = {
    'test_generator': DataGenerator(partition['test'], **params),
    'predition_generator': PredictionDataGenerator(partition['test'], dim=(48,48))
}

# for image_path in partition['test']:
# 	p = os.path.join(params['datapath'], image_path)
# 	img = cv2.imread(p)
# 	if img.shape != (95, 95, 3):
# 		print(img.shape)

# 1/0
checkpoint_path = 'checkpoints/model6-3/cp-0038.ckpt'
model = create_model()
model.load_weights(checkpoint_path)
predictions = []
# for i,p in enumerate(prediction_data):
# 	try:
# 		gen = PredictionDataGenerator(p, dim=(48,48))
# 		res = model.predict_generator(gen, verbose=1)
# 		predictions.append(res)
# 	except:
# 		print("ERROR")
# 		print(i)
# 	break
# predictions = model.predict_generator(generator=data_generators['predition_generator'], verbose=2)
# print(predictions)
for image_path in partition['test']:
	p = os.path.join(params['datapath'], image_path)
	img = cv2.imread(p)
	img = resize(img, 48, True)
	img = img / 255
	img = np.reshape(img, (1, 48, 48, 3))
	pred = model.predict(img)
	predictions.append(pred)
# 	# break

print("SAVING PREDICTIONS")
# print(predictions)
# p = {
#   'predictions': predictions,
#   'list_IDs': data_generators['predition_generator'].list_IDs
# }
with open('predictions/{}.pkl'.format(NAME), 'wb') as f:
    pickle.dump(predictions, f)
