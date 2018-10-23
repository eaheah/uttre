import os
import pandas as pd
import numpy as np

class Attribute:
	def __init__(self, p='/vagrant/imgs/list_attr_celeba.csv'):
		self.df = pd.read_csv(p)

	def mod_image_path(self, image_path):
		if 'png' in image_path:
			return image_path.replace('png', 'jpg')
		return image_path

	def _get_row(self, image_path):
		image_path = self.mod_image_path(image_path)
		return self.df.loc[self.df['image_id'] == image_path]		

	def get_attributes_list(self, image_path):
		row = self._get_row(image_path)
		# print(row.values.tolist())
		i = row.values.tolist()[0][1:]
		l =[[i]]
		n = np.array(l)
		r = n.reshape(len(i), 1, 1)
		return l
	def get_attributes_np(self, image_path):
		row = self._get_row(image_path)
		n = row.values[0][1:]
		print (n.shape)
		return row.values[0][1:]