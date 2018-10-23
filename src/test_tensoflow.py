import tensorflow as tf 
import os
import cv2 
from attributes import Attribute
import numpy as np

p = '/vagrant/imgs/training_data/training_data/aligned'
d = os.listdir(p)

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [95, 95])
    return image_resized

EPOCHS = 10
BATCH_SIZE = 16

filenames = [os.path.join(p, img_path) for img_path in d[:5]]
a = Attribute()
labels = np.array([a.get_attributes_list(img_path) for img_path in d[:5]])
# labels = labels.reshape(labels[0], labels[1], -1, -1)
print (labels.shape)
# labels = tf.constant(l)

features = [_parse_function(img_path) for img_path in d[:5]]
print([feature.shape for feature in features])
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).repeat().batch(BATCH_SIZE)

# dataset = dataset.map(_parse_function)
iterator = dataset.make_one_shot_iterator()
x, y = iterator.get_next()

net = tf.layers.dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)
loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([train_op, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))

    # print(sess.run(next_element))
    # print(sess.run(next_element))
    # print(sess.run(next_element))
    # print(sess.run(next_element))
    # print(sess.run(next_element))


# EPOCHS = 10
# BATCH_SIZE = 16
# # using two numpy arrays
# f = np.random.sample((100,2))
# print(f.shape)
# features, labels = (np.array([np.random.sample((100,2))]), 
#                     np.array([np.random.sample((100,1))]))
# print(features.shape)
# print(labels.shape)
# dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
# iter = dataset.make_one_shot_iterator()
# x, y = iter.get_next()
# # make a simple model
# net = tf.layers.dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
# net = tf.layers.dense(net, 8, activation=tf.tanh)
# prediction = tf.layers.dense(net, 1, activation=tf.tanh)
# loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
# train_op = tf.train.AdamOptimizer().minimize(loss)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(EPOCHS):
#         _, loss_value = sess.run([train_op, loss])
#         print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
