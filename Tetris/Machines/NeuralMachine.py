import numpy as np
from Board import *
from Interfaces.BoardInterface import *
from abc import ABCMeta, abstractmethod
from Machines.Machine import *
import _pickle
from Authen import Env

import os
env = Env()
#if env.use_gpu:
#	os.environ['THEANO_FLAGS']='device=gpu'
#else:
#	os.environ['THEANO_FLAGS'] = 'device=cpu'

from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.utils import np_utils

import theano.tensor as T
from theano import function, pp



class NeuralMachine(Machine):
	def __init__(self):
		super().__init__()
		self.model = None
		
	def build_model(self):
		raise NotImplementedError()
		
	def compile_model(self):
		ada = Adadelta(0.1)
		self.model.compile(
			optimizer=ada,
			loss='categorical_crossentropy',
			metrics=['accuracy']			
			)
		print('Compiled.')

	def load_weights(self):
		self.model.load_weights(self.name)

	def save_weights(self):
		self.model.save_weights(self.name,overwrite=True)

	def load_dataset(self):
		env = Env()
		filenames = os.listdir(env.datafolder)
		
		X=[]
		Y=[]
		for filename in filenames:
			with open(os.path.join(env.datafolder, filename), 'rb') as f:
				Es = _pickle.load(f)
			Xs = [e[0] for e in Es]
			As = [e[1] for e in Es]
			X.append(Xs)
			Y.append(As)
			
		self.X=np.concatenate(X).astype('int32')
		self.Y=np_utils.to_categorical(np.concatenate(Y)).astype('float32')
		
		print(self.X.shape, self.X.dtype)
		print(self.Y.shape, self.Y.dtype)
		print('Data loaded.')
			
class CNNMachine(NeuralMachine):
	def __init__(self):
		super().__init__()
		self.name = 'CNNMachine'		
		self.instantiate()

		self.s = None # theano tensor variable
		self.pi = None # theano tensor variable
		self.params = None # theano shared variables
		
	def instantiate(self):
		self.build_model()
		try:
			self.load_weights()
			print('Weights loaded.')
		except:
			pass

	def Pi(self, state):
		X = self.phi(state)		
		X = X.reshape((1,) + X.shape)
		pi = self.f(X)
		return pi
		
	def build_model(self):
		self.model = Sequential()
		
		self.model.add(Convolution2D(32, 3,3, activation='relu', input_shape=(2,Board.height,Board.width)))
		self.model.add(Convolution2D(32, 3,3, activation='relu'))
		
		self.model.add(Convolution2D(64, 3,3, activation='relu'))
		self.model.add(Convolution2D(64, 3,3, activation='relu'))
		
		self.model.add(Flatten())
		
		self.model.add(Dense(512, activation='relu'))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(5, activation='softmax'))
		
		for layer in self.model.layers:
			print(layer.input_shape,'->',layer.output_shape)
				
		self.s = self.model.get_input_at(0)
		self.pi = self.model.get_output_at(-1)
		self.params = self.model.trainable_weights
		self.f = function([self.s], self.pi[0], allow_input_downcast=True)
		print('Built.')
		
	def train(self):
		self.load_dataset()			
		self.compile_model()
		while True:
			self.model.fit(self.X,self.Y,
				nb_epoch=10,
				verbose=1,
				batch_size=1024,
				shuffle=True
				)
				
			self.save_weights()
