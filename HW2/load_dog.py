import pickle
import numpy as np

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def mini_batch(features,labels,mini_batch_size):
	s = 0
	f = mini_batch_size
	while f < len(labels):
		yield (features[s:f], labels[s:f])
		s = s + mini_batch_size
		f = f + mini_batch_size
        
def test_mini_batch(features,mini_batch_size):
	s = 0
	f = mini_batch_size
	while f < len(features):
		yield features[s:f]
		s = s + mini_batch_size
		f = f + mini_batch_size

def load_training_batch(batch_id,mini_batch_size):
	file_name = 'dc_training_' + str(batch_id) + '.pkl'
	dict = unpickle(file_name)
	features = dict['data']
	labels = dict['labels']
	return mini_batch(features,labels,mini_batch_size)

def load_validation_batch():
	file_name = 'dc_validation.pkl'
	dict = unpickle(file_name)
	features = dict['data']
	labels = dict['labels']
	return features,labels

def load_test_batch(test_batch_id):
	file_name = 'dc_test_' + str(test_batch_id)+ '.pkl'
	dict = unpickle(file_name)
	return dict