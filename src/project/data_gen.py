from tensorflow import keras 
import numpy as np 
import cv2
import pandas as pd
import os
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras - adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
    def __init__(self, list_IDs, batch_size=32, dim=(95,95), n_channels=3, 
                 datapath='/vagrant/imgs/training_data/training_data/aligned',
                 attribute_path='/vagrant/imgs/list_attr_celeba.csv',
                 label_size=40, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.label_size = label_size
        self.datapath = datapath
        self.on_epoch_end()
    
        self.df = pd.read_csv(attribute_path)
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def get_numpy_image(self, image_path):
        img =  cv2.imread(os.path.join(self.datapath, image_path))
        img = cv2.resize(img, self.dim)
        return img / 255
    
    def get_label(self, image_path):
        if 'png' in image_path:
            image_path = image_path.replace('png', 'jpg')
        row = self.df.loc[self.df['image_id'] == image_path]
        label = np.array(row.values.tolist()[0][1:])
        label[label < 0] = 0
        return label

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, self.label_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.get_numpy_image(ID)
            Y[i,] = self.get_label(ID)
            
        return X, Y

class PredictionDataGenerator(DataGenerator):
    def __init__(self, list_IDs, dim=(95,95), n_channels=3, 
                 datapath='/vagrant/imgs/training_data/training_data/aligned', batch_size=200):
        'Initialization'
        self.dim = dim
        print(self.dim)

        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = False
        self.datapath = datapath
        self.on_epoch_end()
        
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.get_numpy_image(ID)
            
        return X