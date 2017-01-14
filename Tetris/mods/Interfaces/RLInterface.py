import numpy as np
import datetime
import _pickle
import time
from functools import reduce

from .BoardInterface import *

from ..Board import *
from ..Authen import *

from theano import function, scan, shared
import theano.tensor as T

	
class PGInterface(BoardInterface):
	def __init__(self, name):
		super().__init__()

		#settings
		self.settings['lr'] = 0.001

		#parameters
		self.previous_R = []
		self.G=0
		
		#gradient estimator
		params = self.machine.params
		Xs = self.machine.s # (H, 2, 22, 10)
		As = T.ivector('As')
		Rs = T.vector('Rs')
		b = T.scalar('b')
		lr = T.scalar('lr')
		pi = self.machine.pi # pi(X ; params) Shape (None,5)
		
		self.Xs_shared = shared(np.zeros((0,0,0,0), dtype='float32'))
		self.As_shared = shared(np.zeros((0,), dtype='int32'))
		self.Rs_shared = shared(np.zeros((0,), dtype='float32'))
		self.lr_shared = shared(np.array(self.settings['lr'] , dtype='float32'))
		self.b_shared = shared(np.array(0., dtype='float32'))
		
		rightsum = Rs.sum() - b
		
		log_pi, _ = scan(lambda p,a : T.log(p[a]), sequences=[pi, As])
		log_pi_sum = log_pi.sum()
		del_params = [rightsum * T.grad(log_pi_sum, param) for param in params]
		updates = [(param, param + lr * del_param) for param, del_param in zip(params, del_params)]
		self.train = function([], [],
			updates=updates,
			givens=[
			(Xs, self.Xs_shared),
			(As, self.As_shared),
			(Rs, self.Rs_shared),
			(b, self.b_shared),
			(lr, self.lr_shared)
			])
		print('Interface ready.')
		
	def initialize(self):
		super().initialize()
		self.experiences = []
		self.G=0
		
	def _learn(self):
		Rs = np.array([e[2] for e in self.experiences], dtype='float32')
		b = np.average(self.previous_R).astype('float32')
		if np.sum(Rs)==b:
			return
		Xs = np.array([e[0] for e in self.experiences], dtype='float32')
		As = np.array([e[1] for e in self.experiences], dtype='int32')
		
		
		self.Xs_shared.set_value(Xs)
		self.As_shared.set_value(As)
		self.Rs_shared.set_value(Rs)
		self.b_shared.set_value(b)
		
		self.train()		
		
	def start(self):
		self.initialize()

		self.isStarted=True
		self.board.S0()
		self.board.nextpiece()
		startTime = time.time()

		while True:
			#feedforward
			self.OnTimer(None)
			#check if it's over
			if self.isOver:
				break
		endTime=time.time()
		if self.G != 0:
			print('Game over. Score : %i, Ticks : %i, Time elapsed : %.1fs'%(self.numLinesRemoved, self.board.t, endTime-startTime))
		self.previous_R.append(self.G)
		if len(self.previous_R) >= 5:
			self.previous_R = self.previous_R[-5:]
		self._learn()
		learn_endTime = time.time()	
		#print('Learned. Time elapsed : %.1fs'%(learn_endTime-endTime))
		return self.G, self.board.t

'''
class DQNRLInterface(RLInterface):
	def __init__(self, name, machine):
		super().__init__()
		self.machine = machine
		
	def OnTick(self):
		#choose e-greedy action!		
		use_random = np.random.binomial(1, self.ep, 1)[0]	
		if use_random:
			pi = [0,0,0,0,0]
			pi[np.random.randint(5)] = 1
		else:
			pi = self.machine.Pi(self.board.S())
		
		if pi is not None:
			self.perform_action(np.random.choice(5, p=pi))
		
	def perform_action(self, action):
		self.actionhistory.append((self.board.t, action))

		s = self.board.phi()

		def action2A(self, action):
			result = [0, 0, 0, 0, 0]
			result[action] = 1
			return result

		a = action2A(self, action)
		r = self.board.T(action)
		sprime = self.board.phi()

		e = (s,a,r,sprime)

		self._add_E(e)

	def _learn(self):
		# Es = [e1,e2,e3,...,]
		# e = (s,a,r,s')
		Es = self._pick_Es(self.train_batch)
		Ss = np.array([E[0] for E in Es])
		As = np.array([E[1] for E in Es])
		Rs = np.array([E[2] for E in Es])
		Sps = np.array([E[3] for E in Es])
		
		#we have an issue here.
		#the model is function approximator of Q function.
		#f -> [-inf, inf]
		#currently, f -> [0,1]
		
		#1. get Q
		Qs = self.machine.model.predict(
			Ss,
			verbose=0,
			batch_size=self.train_batch)

		#2. get max Q(s',a')
		Q_of_sp_aps = self.machine.model.predict(
			Sps,
			verbose=0,
			batch_size=self.train_batch)			
		maxQ_of_sps = np.max(Q_of_sp_aps, axis=1)
		
		#3. set target_Q		
		target_Qs = Qs.copy()
		actions = np.nonzero(As)		
		target_Qs[actions] = Rs + self.gamma * maxQ_of_sps
					
		self.machine.model.fit(
			Ss, target_Qs,
			nb_epoch = 1,
			verbose=0,
			batch_size=self.train_batch,
			shuffle=True)
			
		#4. print
		#new_Qs = self.machine.model.predict(
		#	Ss,
		#	verbose=0,
		#	batch_size=self.train_batch)
			
		#for a,b,c in zip(Qs.flatten(), target_Qs.flatten(), new_Qs.flatten()):
		#	print(a, b, c)
		#a=input("go")
		
		#print('learned')

	def start(self):
		self.initialize()

		self.isStarted=True
		self.board.S0()
		self.board.nextpiece()
		startTime = time.time()

		while True:
			self.exp_count+=1

			#feedforward
			self.OnTimer(None)			

			#check if it's over
			if self.isOver:
				break

			#learn
			if self.exp_count % self.train_interval == 0:
				self._learn()

		self.machine.save_weights()
		endTime=time.time()
		print('Game over. Score : %i, Ticks : %i, Time elapsed : %.3f'%(self.numLinesRemoved, self.board.t, endTime-startTime))
		return self.numLinesRemoved, self.board.t
<<<<<<< HEAD

class PGInterface(RLInterface):
	def __init__(self, name, machine):
		super().__init__(maxTick=10000)
		self.name = 'PGBoard'
		self.machine = machine
		self.lr = 0.0001
		self.previous_R = []
		self.G=0
		
		params = self.machine.params
		Xs = self.machine.s # (H, 2, 22, 10)
		As = T.ivector('As')
		Rs = T.vector('Rs')
		b = T.scalar('b')
		lr = T.scalar('lr')
		pi = self.machine.pi # pi(X ; params) Shape (None,5)
		
		self.Xs_shared = shared(np.zeros((0,0,0,0), dtype='float32'))
		self.As_shared = shared(np.zeros((0,), dtype='int32'))
		self.Rs_shared = shared(np.zeros((0,), dtype='float32'))
		self.lr_shared = shared(np.array(self.lr , dtype='float32'))
		self.b_shared = shared(np.array(0., dtype='float32'))
		
		rightsum = Rs.sum() - b
		
		log_pi, _ = scan(lambda p,a : T.log(p[a]), sequences=[pi, As])
		log_pi_sum = log_pi.sum()
		del_params = [rightsum * T.grad(log_pi_sum, param) for param in params]
		updates = [(param, param + lr * del_param) for param, del_param in zip(params, del_params)]
		self.train = function([], [],
			updates=updates,
			givens=[
			(Xs, self.Xs_shared),
			(As, self.As_shared),
			(Rs, self.Rs_shared),
			(b, self.b_shared),
			(lr, self.lr_shared)
			])
		print('Interface ready.')
		
	def initialize(self):
		super().initialize()
		self.experiences = []
		self.G=0
		
	def OnTick(self):
		pi = self.machine.Pi(self.board.S())
		#print(self.phi(self.board.S()))
		#print(pi)
		#input('Go?')
		if pi is not None:
			self.perform_action(np.random.choice(5, p=pi))
		
	def perform_action(self, action):
		s = self.phi(self.board.S()).astype('float32')
		a = action
		r = self.board.T(action)
		e = (s, a, r)
		self.experiences.append(e)
		self.G += r

	@abstractmethod
	def _learn(self):
		Rs = np.array([e[2] for e in self.experiences], dtype='float32')
		b = np.average(self.previous_R).astype('float32')
		if np.sum(Rs)==b:
			return
		Xs = np.array([e[0] for e in self.experiences], dtype='float32')
		As = np.array([e[1] for e in self.experiences], dtype='int32')
		
		
		self.Xs_shared.set_value(Xs)
		self.As_shared.set_value(As)
		self.Rs_shared.set_value(Rs)
		self.b_shared.set_value(b)
		
		self.train()		
		
	def start(self):
		self.initialize()

		self.isStarted=True
		self.board.S0()
		self.board.nextpiece()
		startTime = time.time()

		while True:
			#feedforward
			self.OnTimer(None)
			#check if it's over
			if self.isOver:
				break
		endTime=time.time()
		if self.G != 0:
			print('Game over. Score : %i, Ticks : %i, Time elapsed : %.1fs'%(self.numLinesRemoved, self.board.t, endTime-startTime))
		self.previous_R.append(self.G)
		if len(self.previous_R) >= 5:
			self.previous_R = self.previous_R[-5:]
		self._learn()
		learn_endTime = time.time()	
		#print('Learned. Time elapsed : %.1fs'%(learn_endTime-endTime))
		return self.G, self.board.t
=======
'''