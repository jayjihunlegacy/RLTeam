import numpy as np
import _pickle
from abc import ABCMeta, abstractmethod
import os

import theano.tensor as T
import theano

from keras.layers import Dense, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adadelta
from keras.utils import np_utils

from .Machine import *
from ..Interfaces.BoardInterface import *
from ..Board import *
from ..Authen import Env

class NeuralMachine(Machine):
	__metaclass__=ABCMeta
	"""
	:class:`NeuralMachine` is an abstract class of machines with Neural Network.\
	It has to implement :meth:`build_model` which specifies model architecture.

	Since NeuralMachine typically uses GPU, only one process is used for evaluation.

	:param int batch_size: size of batch to be used during training
	"""
	def __init__(self, name, batch_size=128):
		super().__init__()
		self.model = None
		"""Neural network for machine."""

		self.f = None
		""":class:`theano.function` instance which is called in :meth:`Pi`. \
		Takes (1, 2, h, w) tensor and outputs (5,) tensor."""

		self.S = None
		""":class:`theano.tensor` variable which is an input of neural network. \
		It is a (None, 2, h, w) tensor."""
		
		self.P = None
		""":class:`theano.tensor` variable which is an output of neural network. \
		It can be policy for :class:`CNNPolicyMachine` or value for :class:`CNNValueMachine`."""

		self.pi = None
		""":class:`theano.tensor` variable which is an output policy of neural network. \
		For PolicyMachine, :attr:`pi` and :attr:`P` can be identical."""

		self.params = None
		""":class:`list` variable which is a list of trainable parameter of neural networks."""

		self.batch_size = batch_size
		"""Size of batch to be used during training."""

		self.name = name
		"""Name of machine. Used to save weights."""
		
	evaluate=Machine.sequential_evaluate

	@abstractmethod
	def build_model(self):
		"""
		:meth:`build_model` specifies model architecture.
		"""
		raise NotImplementedError()
		
	def compile_model(self, lr = 0.01):
		"""
		:meth:`compile_model` is a widely used compilation function.\
		This procedure is necessary only when Keras fit function is used.

		:param float lr: learning rate of Adadelta optimizer
		"""
		ada = Adadelta(lr)
		self.model.compile(
			optimizer=ada,
			loss='categorical_crossentropy',
			metrics=['accuracy']			
			)
		print('Compiled.')

	def load_weights(self):
		"""
		:meth:`load_weights` loads weight from binary file.\
		File name is attained from its class name and architecture.

		For example, 'CNNPolicyMachine_20x4'		
		"""
		self.model.load_weights(self.name+'_%ix%i'%(self.height, self.width))

	def save_weights(self):
		"""
		:meth:`save_weights` save weight to binary file.\
		File name is attained from its class name and architecture.

		For example, 'CNNPolicyMachine_20x4'		
		"""
		self.model.save_weights(self.name+'_%ix%i'%(self.height, self.width),overwrite=True)

	def Pi(self, state):
		"""
		:meth:`Pi` returns a probability distribution of actions given *state*. \
		For :class:`NeuralMachine`, this is done by calling :attr:`f`.

		:param tuple state: board state in a tuple form. \
		See :meth:`Tetris.mods.Board.Board.S` for details.

		:returns pi: probability distribution of actions. must sum to 1.
		:rtype: list
		"""
		X = self.phi(state)		
		X = X.reshape((1,) + X.shape)
		pi = self.f(X)
		return pi

	def load_dataset(self):
		"""
		:meth:`load_dataset` loads **trajectory** data. Trajectory is a tuple of \
		(S, A, R, S'). Thus, it sets **X**, **A**, **Y**, **R**, **Xp**.

		1. **X** : (None, 2, h, w), float32
		2. **A** : (None, ), int32
		3. **Y** : (None, 5), float32
		4. **R** : (None, ), float32
		5. **Xp** : (None, 2, h, w), float32
		"""
		env = Env()
		filenames = os.listdir(env.datafolder)
		
		X=[]
		Y=[]
		R=[]
		Xp=[]
		for filename in filenames:
			with open(os.path.join(env.datafolder, filename), 'rb') as f:
				Es = _pickle.load(f)
			Xs = [e[0] for e in Es]
			As = [e[1] for e in Es]
			Rs = [e[2] for e in Es]
			Xps = [e[3] for e in Es]

			X.append(Xs)
			Y.append(As)
			R.append(Rs)
			Xp.append(Xps)
			
		self.X=np.concatenate(X).astype('float32')
		self.A = np.concatenate(Y).astype('int32')
		self.Y=np_utils.to_categorical(np.concatenate(Y)).astype('float32')
		self.R=np.concatenate(R).astype('float32')
		self.Xp=np.concatenate(Xp).astype('float32')
		
		print('Data loaded. Amount : %i.'%(self.X.shape[0]))
					
class CNNPolicyMachine(NeuralMachine):
	"""
	:class:`CNNPolicyMachine` is machine with CNN based **Policy approximating** machine.

	:param str name: name of machine
	:param int batch_size: size of batch to be used during training
	"""
	def __init__(self, name='CNNPolicyMachine', batch_size=2048):
		super().__init__(
			name = name,
			batch_size=batch_size
			)		
		self.instantiate()

	def instantiate(self):
		"""
		:meth:`instantiate` instantiates neural network. It builds model and set tensor parameters \
		including :attr:`S`, :attr:`P`, :attr:`pi`, and :attr:`params`. \
		Also, the function :attr:`f` is compiled. \
		After building model, it loads model weights.
		"""
		self.model = self.build_model()

		self.model.summary()
		self.S = self.model.get_input_at(0)
		self.P = self.model.get_output_at(0)
		self.params = self.model.trainable_weights
		self.pi = self.P		
		
		self.f = theano.function([self.S], self.pi[0], allow_input_downcast=True)

		print('Built.')

		try:
			self.load_weights()
			print('Weights loaded.')
		except:
			pass
		
	def build_model(self):
		"""
		:meth:`build_model` specified model architecture.

		:returns model: keras model
		:rtype: :class:`keras.models`		
		"""
		model = Sequential()		
		model.add(Convolution2D(32, 3,3, activation='relu', border_mode='same', input_shape=(2,Board.height,Board.width)))
		
		model.add(Flatten())
		
		model.add(Dense(128, activation='relu'))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(5, activation='softmax'))
		
		return model
		
	def train(self, nb_epoch=10):
		"""
		:meth:`train` is a special training function for :class:`CNNPolicyMachine`.

		Since :class:`CNNPolicyMachine` has a policy-approximating model, train the\
		model just like solving a **classification** problem.

		:param int nb_epoch: number of epoch
		"""
		self.load_dataset()			
		self.compile_model()
		while True:
			self.model.fit(self.X,self.Y,
				nb_epoch=nb_epoch,
				verbose=1,
				batch_size=self.batch_size,
				shuffle=True
				)
				
			self.save_weights()
				
class CNNValueMachine(NeuralMachine):
	"""
	:class:`CNNValueMachine` is machine with CNN based **Value approximating** machine.

	:param str name: name of machine
	:param int batch_size: size of batch to be used during training
	:param str v_to_pi: way to get pi from Value. ('greedy' or 'softmax' or 'e-greedy')
	:param float epsilon: epsilon value to be used when *v_to_pi* is 'e-greedy'
	"""
	def __init__(self, name='CNNValueMachine', batch_size=1024, v_to_pi='softmax', epsilon=.1):
		super().__init__(
			name=name, 
			batch_size=batch_size
			)

		self.q = None
		""":class:`theano.function` instance which calculates Q of actions. \
		Takes (None, 2, h, w) tensor and outputs (None, 5) tensor."""

		self.v_to_pi = v_to_pi
		"""Way to get pi from Value. ('greedy' or 'softmax' or 'e-greedy')"""

		self.epsilon = epsilon
		"""Epsilon value to be used when :attr:`v_to_pi` is 'e-greedy'"""

		self.instantiate()

	def instantiate(self):
		"""
		:meth:`instantiate` instantiates neural network. It builds model and set tensor parameters \
		including :attr:`S`, :attr:`P`, :attr:`pi`, and :attr:`params`. \
		Also, the function :attr:`f` and :attr:`q` are compiled. \
		After building model, it loads model weights.
		"""
		self.model = self.build_model()
		self.model.summary()
		
		self.S = self.model.get_input_at(0)
		self.P = self.model.get_output_at(0)
		self.params = self.model.trainable_weights

		#define pi respect to Q. (max or softmax)
		self.pi = self.get_pi_from_v(self.P)
		
		self.f = theano.function([self.S], self.pi[0], allow_input_downcast=True)
		self.q = theano.function([self.S], self.P, allow_input_downcast=True)

		print('Built.')

		try:
			self.load_weights()
			print('Weights loaded.')
		except:
			pass
			
	def build_model(self):
		"""
		:meth:`build_model` specified model architecture.

		:returns model: keras model
		:rtype: :class:`keras.models`		
		"""
		model = Sequential()		
		model.add(Convolution2D(32, 3,3, activation='relu', border_mode='same', input_shape=(2,Board.height,Board.width)))
		
		model.add(Flatten())
		
		model.add(Dense(128, activation='relu'))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(5, activation='linear'))
		return model

	def get_pi_from_v(self, Q):
		if self.v_to_pi == 'greedy' or self.v_to_pi == 'e-greedy':
			greedy_actions = T.argmax(Q, axis=-1)
			greedy_pi = T.extra_ops.to_one_hot(greedy_actions, nb_class=5, dtype='int32')

			if self.v_to_pi=='greedy':
				return greedy_pi
			else:
				return T.fill(Q, self.epsilon/5) + (1-self.epsilon) * greedy_pi
		elif self.v_to_pi == 'softmax':
			return T.nnet.softmax(Q)
		else:
			raise Exception()

class CNNActorCriticMachine(NeuralMachine):
	"""
	:class:`CNNActorCriticMachine` is machine with CNN based actor-critic machine. \
	Actor is a model which approximates **Policy**, while \
	critic is a model which approximates **Value**.

	:param str name: name of machine
	:param int batch_size: size of batch to be used during training
	"""
	def __init__(self, name='CNNActorCriticMachine', batch_size=2048):
		super().__init__(
			name = name,
			batch_size=batch_size
			)		

		self.S2 = None
		""":class:`theano.tensor` variable which is an input of neural network. \
		It is a (None, 2, h, w) tensor."""

		self.P2 = None
		""":class:`theano.tensor` variable which is an output of neural network. \
		It can be policy for :class:`CNNPolicyMachine` or value for :class:`CNNValueMachine`."""
		self.instantiate()

	def instantiate(self):
		"""
		:meth:`instantiate` instantiates neural network. It builds model and set tensor parameters \
		including :attr:`S`, :attr:`P`, :attr:`pi`, and :attr:`params`. \
		Also, the function :attr:`f` and :attr:`q` are compiled. \
		After building model, it loads model weights.
		"""
		self.model = self.build_model()
		self.model.summary()

		#2. Value Function (state value function)
		self.V = self.S
		for layer in self.model.layers[:-1]:
			self.P2 = layer(self.P2)
		self.P2 = Dense(1, activation='linear')(self.P2)
		self.criticmodel = Model(input=self.S, output=self.P2)
				
		self.S = self.model.get_input_at(0)
		self.P = self.model.get_output_at(0)
		self.params = self.model.trainable_weights
		self.criticparams = self.criticmodel.trainable_weights
		self.pi = self.P		
		
		self.f = theano.function([self.S], self.pi[0], allow_input_downcast=True)
		print('Built.')

		try:
			self.load_weights()
			print('Weights loaded.')
		except:
			pass			
		
	def build_model(self):
		"""
		:meth:`build_model` specified model architecture.

		:attr:`S`, :attr:`P`, :attr:`pi`, and :attr:`params` are set.
		:attr:`f` is compiled.
		"""
		#1. Policy Function
		model = Sequential()		
		model.add(Convolution2D(32, 3,3, activation='relu', border_mode='same', input_shape=(2,Board.height,Board.width)))
		
		model.add(Flatten())
		
		model.add(Dense(128, activation='relu'))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(5, activation='softmax'))
		
		return model