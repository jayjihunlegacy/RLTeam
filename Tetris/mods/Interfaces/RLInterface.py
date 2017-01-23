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

from keras.layers import Input
	
class PGInterface(BoardInterface):
	"""
	:class:`PGInterface` is an interface that can run policy gradient algorithms. \
	It runs several policy gradient algorithms on its :attr:`machine`.

	:param machine: machine to play and be trained
	:type machine: :class:`Tetris.mods.Machine.Machine`
	:param str pgmode: type of Policy Gradient algorithms. 'vanilla', 'pgt', \
	and 'actorcritic' are available.
	:param float lr: learning rate
	"""
	def __init__(self, machine, pgmode, lr=0.0001, **kwargs):
		super().__init__(machine=machine, **kwargs)

		#settings
		self.settings['lr'] = lr
		self.settings['pgmode'] = pgmode #pgmode 'vanilla', 'pgt', 'actorcritic'
		self.settings['name'] = 'PGInterface(%s)'%(pgmode)
		self.settings['collect_traj'] = True	

		#parameters
		self.Gs = []
		"""Hisotry of Return throughout training."""
		self.Hs = []
		"""Hisotry of horizon length throughout training."""
		self.learn = None
		"""Learning function. Learns by calling :attr:`train_core`."""
		self.train_core = None
		"""Core training theano function."""

		#build trainer
		if self.settings['pgmode'] == 'vanilla':
			self.build_trainer_vanilla()
		elif self.settings['pgmode'] == 'pgt':
			self.build_trainer_pgt()
		elif self.settings['pgmode'] == 'actorcritic':
			self.build_trainer_actorcritic()

		print('Interface ready.')

	def build_trainer_vanilla(self):
		"""
		:meth:`build_trainer_vanilla` compiles a :attr:`train_core` theano function which implements \
		**Vanilla Policy Gradient algorithm**, and algorithm-specific :attr:`learn` function.
		"""
		#gradient estimator
		params = self.machine.params
		Ss = self.machine.S		# (H, 2, h, w)
		pi = self.machine.pi	# (H, 5)
		As = T.ivector('As')	# (H,)
		Rs = T.vector('Rs')		# (H,)
		b = T.scalar('b')		# scalar
		index = T.iscalar('index')
		
		#update formula
		resp_R = Rs.sum()-b
		log_pi, _ = scan(lambda t, a, Pi : T.log(Pi[t][a]), \
			sequences=[T.arange(As.shape[0]), As], non_sequences=pi)
		sum_log_pi = log_pi.sum()
		gradients = [T.grad(sum_log_pi, param) * resp_R for param in params]
		updates = [(param, param + self.settings['lr'] * del_param) \
			for param, del_param in zip(params, gradients)]

		#shared variables.
		self.Ss_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)
		self.As_shared = shared(np.zeros((0,), dtype='int32'))						# (H, )
		self.Rs_shared = shared(np.zeros((0,), dtype='float32'))					# (H, )
		self.b_shared = shared(np.array(0., dtype='float32'))						# scalar
		
		#index as input + shared data => transfer only index when batch is many
		self.train_core = function([index], [],
			updates=updates,
			givens=[
			(Ss, self.Ss_shared[index * self.machine.batch_size : (index+1) * self.machine.batch_size]),
			(As, self.As_shared[index * self.machine.batch_size : (index+1) * self.machine.batch_size]),
			(Rs, self.Rs_shared[index * self.machine.batch_size : (index+1) * self.machine.batch_size]),
			(b, self.b_shared)
			])

		def learn():
			Rs = np.array([e[2] for e in self.trajectory], dtype='float32')
			H = len(Rs)
			b = np.mean(self.previous_G).astype('float32')
			Ss = np.array([e[0] for e in self.trajectory], dtype='float32')
			As = np.array([e[1] for e in self.trajectory], dtype='int32')

			self.Ss_shared.set_value(Ss)
			self.As_shared.set_value(As)
			self.Rs_shared.set_value(Rs)
			self.b_shared.set_value(b)		
		
			for i in range(-(-H//self.machine.batch_size)):
				self.train_core(i)
		self.learn = learn

	def build_trainer_pgt(self):
		"""
		:meth:`build_trainer_pgt` compiles a :attr:`train_core` theano function which implements \
		**PGT Policy Gradient algorithm**, and algorithm-specific :attr:`learn` function.
		"""
		#gradient estimator
		params = self.machine.params
		Ss = self.machine.S		# (H, 2, h, w)
		pi = self.machine.pi	# (H, 5)
		As = T.ivector('As')	# (H,)
		Rs = T.vector('Rs')		# (H,)
		index = T.iscalar('index')
		
		#update formula
		resp_Rs, _ = scan(lambda t, Rs : T.sum(Rs[t:]), 
			sequences=T.arange(Rs.shape[0]), non_sequences=Rs)
		log_pi, _ = scan(lambda t, a, resp_R, Pi : T.log(Pi[t][a]) * resp_R, \
			sequences=[T.arange(As.shape[0]), As, resp_Rs], non_sequences=pi)
		sum_log_pi = log_pi.sum()

		objective = sum_log_pi

		#weight regularizer
		for param in params:
			objective -= 0.0000001 * T.sum(param)

		gradients = [T.grad(objective, param) for param in params]
		updates = [(param, param + self.settings['lr'] * del_param) \
			for param, del_param in zip(params, gradients)]

		#shared variables.
		self.Ss_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)
		self.As_shared = shared(np.zeros((0,), dtype='int32'))						# (H, )
		self.Rs_shared = shared(np.zeros((0,), dtype='float32'))					# (H, )
		
		#index as input + shared data => transfer only index when batch is many
		self.train_core = function([index], [],
			updates=updates,
			givens=[
			(Ss, self.Ss_shared[index * self.machine.batch_size : (index+1) * self.machine.batch_size]),
			(As, self.As_shared[index * self.machine.batch_size : (index+1) * self.machine.batch_size]),
			(Rs, self.Rs_shared[index * self.machine.batch_size : (index+1) * self.machine.batch_size])
			])

		def learn():
			Rs = np.array([e[2] for e in self.trajectory], dtype='float32')
			H = len(Rs)
			Ss = np.array([e[0] for e in self.trajectory], dtype='float32')
			As = np.array([e[1] for e in self.trajectory], dtype='int32')

			self.Ss_shared.set_value(Ss)
			self.As_shared.set_value(As)
			self.Rs_shared.set_value(Rs)
		
			for i in range(-(-H//self.machine.batch_size)):
				self.train_core(i)
		self.learn = learn

	def build_trainer_actorcritic(self):
		"""
		**NOT IMPLEMENTED**
		"""
		#first, use MC!!

		#gradient estimator
		params = self.machine.params
		criticparams = self.machine.criticparams
		Ss = self.machine.S		# (H, 2, h, w)
		pi = self.machine.pi	# (H, 5)
		Vs = self.machine.V		# (H, 1)
		As = T.ivector('As')	# (H,)
		Rs = T.vector('Rs')		# (H,)
		index = T.iscalar('index')
		
		#update formula
		resp_Rs, _ = scan(lambda t, Rs : T.sum(Rs[t:]), 
			sequences=T.arange(Rs.shape[0]), non_sequences=Rs)
		log_pi, _ = scan(lambda t, a, resp_R, Pi : T.log(Pi[t][a]) * resp_R, \
			sequences=[T.arange(As.shape[0]), As, resp_Rs], non_sequences=pi)
		sum_log_pi = log_pi.sum()

		objective = sum_log_pi

		#add Vs learning formula!!


		#weight regularizer
		for param in params:
			objective -= 0.0000001 * T.sum(param)

		gradients = [T.grad(objective, param) for param in params]
		updates = [(param, param + self.settings['lr'] * del_param) \
			for param, del_param in zip(params, gradients)]

		#shared variables.
		self.Ss_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)
		self.As_shared = shared(np.zeros((0,), dtype='int32'))						# (H, )
		self.Rs_shared = shared(np.zeros((0,), dtype='float32'))					# (H, )
		
		#index as input + shared data => transfer only index when batch is many
		self.train_core = function([index], [],
			updates=updates,
			givens=[
			(Ss, self.Ss_shared[index * self.machine.batch_size : (index+1) * self.machine.batch_size]),
			(As, self.As_shared[index * self.machine.batch_size : (index+1) * self.machine.batch_size]),
			(Rs, self.Rs_shared[index * self.machine.batch_size : (index+1) * self.machine.batch_size])
			])

		def learn():
			Rs = np.array([e[2] for e in self.trajectory], dtype='float32')
			H = len(Rs)
			Ss = np.array([e[0] for e in self.trajectory], dtype='float32')
			As = np.array([e[1] for e in self.trajectory], dtype='int32')

			self.Ss_shared.set_value(Ss)
			self.As_shared.set_value(As)
			self.Rs_shared.set_value(Rs)
		
			for i in range(-(-H//self.machine.batch_size)):
				self.train_core(i)
		self.learn = learn

	def train(self):
		"""
		:meth:`train` is a top-level train function. \
		Repeatedly calls :meth:`start` to produce trajectories and learn.

		Every 100 episode, it saves history and model weights.
		"""
		historyname = '%s_%s_%ix%i.pkl'% \
		(self.settings['name'], self.machine.name, self.board.height, self.board.width)

		try:
			with open(historyname, 'rb') as f:
				history = _pickle.load(f)
		except:
			# G for return, and H for horizon
			history = {'G' : [], 'H' : []}

		self.Gs = history['G']
		self.Hs = history['H']

		episode_num = len(history['G'])
		print('Start Policy Gradient.')
		firsttime = time.time()
		starttime = firsttime
		Gs_100, Hs_100 = [], []
		try:
			while True:
				G, H = self.start()
				self.Gs.append(G)
				self.Hs.append(H)

				Gs_100.append(G)
				Hs_100.append(H)

				episode_num += 1

				if episode_num % 100 == 0:
					nowtime = time.time()
					print('Episode #%i. Mean G : %.2f(%.2f/%.2f), Mean ticks : %.2f(%i/%i). Took %.1fs, elapsed %.1fs'\
						%(episode_num, \
						np.mean(Gs_100), np.min(Gs_100), np.max(Gs_100), \
						np.mean(Hs_100), np.min(Hs_100), np.max(Hs_100), \
						nowtime-starttime, nowtime-firsttime))

					self.machine.save_weights()
					starttime = nowtime
					with open(historyname, 'wb') as f:
						_pickle.dump(history, f)

					Gs_100, Hs_100 = [], []
		except KeyboardInterrupt:
			with open(historyname, 'wb') as f:
				_pickle.dump(history, f)
		
	def start(self):
		"""
		:meth:`start` runs one episode and produces one trajectory.

		After each episode is done, it calls :attr:`learn` function to learn.

		:returns (G, H): (Return, Horizon)
		:rtype: (float, int)
		"""
		self.initialize()
		self.isStarted=True

		startTime = time.time()
		while True:
			self.tick(None)
			if self.isOver:
				break

		endTime=time.time()

		self.learn()
		learn_endTime = time.time()	

		return self.G, self.board.t

class ValueRLInterface(BoardInterface):
	"""
	:class:`ValueRLInterface` is an interface that can run value-based learning algorithsm. \
	It runs several learning algorithms on its :attr:`machine`.

	Value-Based learning algorithms are all basically forcing Bellman Equation on visited states.\
	Available training algorithms are:

	1. **DQN**
	2. **Q-Learning**
	3. **SARSA**
	4. **Expected-SARSA**

	:param machine: machine to play and be trained
	:type machine: :class:`Tetris.mods.Machine.Machine`
	:param str rlmode: type of learning algorithms. \
	'q-learning', 'dqn', 'double-dqn', 'sarsa', and 'expected-sarsa' are available.
	:param float lr: learning rate
	"""
	def __init__(self, machine, rlmode, lr=0.0001, **kwargs):
		super().__init__(machine=machine, **kwargs)

		#settings
		self.settings['lr'] = lr
		self.settings['rlmode'] = rlmode
		self.settings['name'] = 'RLInterface(%s)'%(rlmode)
		self.settings['collect_traj'] = True
		self.settings['refresh_traj'] = False

		#parameters
		self.Gs = []
		"""Hisotry of Return throughout training."""
		self.Hs = []
		"""Hisotry of horizon length throughout training."""
		self.Costs = []
		"""Hisotry of Value-Approximation costs throughout training."""
		self.learn = None
		"""Learning function. Learns by calling :attr:`train_core`."""
		self.train_core = None
		"""Core training theano function."""

		#build trainer
		if self.settings['rlmode'] == 'q-learning':
			self.build_trainer_qlearning()
		elif self.settings['rlmode'] == 'dqn':
			self.build_trainer_dqn()
		elif self.settings['rlmode'] == 'double-dqn':
			self.build_trainer_doubledqn()
		elif self.settings['rlmode'] == 'sarsa':
			self.build_trainer_sarsa()
		elif self.settings['rlmode'] == 'expected-sarsa':
			self.build_trainer_expectedsarsa()

		print('Interface ready.')

	def _get_another_output(self, input, model):
		tmp = input
		for layer in model.layers:
			tmp = layer(tmp)
		return tmp

	def _backup(self, approxs, targets, params):
		cost = T.square(approxs - targets).mean()

		gradients = T.grad(cost, params)
		updates = [(param, param-self.settings['lr'] * g) \
			for param, g in zip(params, gradients)]

		return cost, updates

	def build_trainer_dqn(self):
		"""
		:meth:`build_trainer_dqn` compiles a :attr:`train_core` theano function which implements \
		**DQN**, and algorithm-specific :attr:`learn` function.
		"""
		params = self.machine.params
		index = T.iscalar('index')
		Ss = self.machine.S			# (H, 2, h, w)
		As = T.ivector('As')		# (H,)
		Rs = T.vector('Rs')			# (H,)
		Qsof_S = self.machine.P

		model2 = self.machine.get_model()
		params2 = model2.trainable_weights

		#initialize, copy weights
		for param, param2 in zip(params, params2):
			param.set_value(param2.get_value())
		Sps = model2.get_input_at(0)
		Qsof_Sp_Ap = model2.get_output_at(0)

		#update formula
		targets = Rs + T.max(Qsof_Sp_Ap, axis=-1)

		Qsof_S_A, _ = scan(lambda i, A, P: P[i,A], \
			sequences=[T.arange(As.shape[0]), As], \
			non_sequences=Qsof_S)

		cost, updates = self._backup(Qsof_S_A, targets, params)

		#shared variables.
		self.Ss_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)
		self.As_shared = shared(np.zeros((0,), dtype='int32'))						# (H, )
		self.Rs_shared = shared(np.zeros((0,), dtype='float32'))					# (H, )
		self.Sps_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)

		self.train_core = function([index], [cost],
			updates=updates,
			givens=[
				(Ss, self.Ss_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(As, self.As_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(Rs, self.Rs_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(Sps, self.Sps_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size])
			])

		def learn():
			"""
			Since Q-learning is off-policy, we can use all the experiences accumulated before.
			"""
			replace_rate = 0.01
			N = 8192*2
			if np.random.binomial(1, replace_rate):
				for param, param2 in zip(params, params2):
					param.set_value(param2.get_value())

			Ss = np.array([e[0] for e in self.trajectory], dtype='float32')
			As = np.array([e[1] for e in self.trajectory], dtype='int32')
			Rs = np.array([e[2] for e in self.trajectory], dtype='float32')
			Sps = np.array([e[3] for e in self.trajectory], dtype='float32')
			n = len(Rs)

			if n >= N:
				ind = np.random.choice(n, size=(N,))
				Ss = Ss[ind]
				As = As[ind]
				Rs = Rs[ind]
				Sps = Sps[ind]

			self.Ss_shared.set_value(Ss)
			self.As_shared.set_value(As)
			self.Rs_shared.set_value(Rs)
			self.Sps_shared.set_value(Sps)
			
			costs = []
			for i in range(-(-n//self.machine.batch_size)):
				costs.append(self.train_core(i))
			return np.mean(costs)

		self.learn=learn

	def build_trainer_qlearning(self):
		"""
		:meth:`build_trainer_qlearning` compiles a :attr:`train_core` theano function which implements \
		**Q-Learning**, and algorithm-specific :attr:`learn` function.
		"""
		params = self.machine.params
		index = T.iscalar('index')
		Ss = self.machine.S			# (H, 2, h, w)
		As = T.ivector('As')		# (H,)
		Rs = T.vector('Rs')			# (H,)
		Qsof_S = self.machine.P
		Sps = Input(shape=(2, Board.height, Board.width)) # (H, 2, h, w)
		Qsof_Sp_Ap = self._get_another_output(Sps, self.machine.model)

		#update formula
		targets = Rs + T.concatenate([T.max(Qsof_Sp_Ap[:-1], axis=-1), T.constant(0)])

		Qsof_S_A, _ = scan(lambda i, A, P: P[i,A], \
			sequences=[T.arange(As.shape[0]), As], \
			non_sequences=Qsof_S)

		cost, updates = self._backup(Qsof_S_A, targets, params)

		#shared variables.
		self.Ss_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)
		self.As_shared = shared(np.zeros((0,), dtype='int32'))						# (H, )
		self.Rs_shared = shared(np.zeros((0,), dtype='float32'))					# (H, )
		self.Sps_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)

		self.train_core = function([index], [cost],
			updates=updates,
			givens=[
				(Ss, self.Ss_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(As, self.As_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(Rs, self.Rs_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(Sps, self.Sps_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size])
			])

		def learn():
			"""
			Since Q-learning is off-policy, we can use all the experiences accumulated before.
			"""
			N = 8192 * 2
			Ss = np.array([e[0] for e in self.trajectory], dtype='float32')
			As = np.array([e[1] for e in self.trajectory], dtype='int32')
			Rs = np.array([e[2] for e in self.trajectory], dtype='float32')
			Sps = np.array([e[3] for e in self.trajectory], dtype='float32')
			n = len(Rs)

			if n >= N:
				ind = np.random.choice(n, size=(N,))
				Ss = Ss[ind]
				As = As[ind]
				Rs = Rs[ind]
				Sps = Sps[ind]

			self.Ss_shared.set_value(Ss)
			self.As_shared.set_value(As)
			self.Rs_shared.set_value(Rs)
			self.Sps_shared.set_value(Sps)
			
			costs = []
			for i in range(-(-n//self.machine.batch_size)):
				costs.append(self.train_core(i))
			return np.mean(costs)

		self.learn=learn

	def build_trainer_sarsa(self):
		"""
		:meth:`build_trainer_sarsa` compiles a :attr:`train_core` theano function which implements \
		**SARSA**, and algorithm-specific :attr:`learn` function.
		"""
		self.settings['refresh_traj'] = True
		#Since SARSA is an on-policy learning algorithm, trajectories from former pi cannot be used.

		params = self.machine.params
		index = T.iscalar('index')
		Ss = self.machine.S			# (H, 2, h, w)
		As = T.ivector('As')		# (H,)
		Rs = T.vector('Rs')			# (H,)
		Qsof_S = self.machine.P
		Sps = Input(shape=(2, Board.height, Board.width)) # (H, 2, h, w)
		Qsof_Sp = self._get_another_output(Sps, self.machine.model)

		#update formula
		#Vsof_Sp = T.max(Qsof_Sp_Ap, axis=-1)
		Qsof_Sp_Ap, _ = scan(lambda t, A, P: P[t,A], \
			sequences=[T.arange(As.shape[0]-1), As[1:]], \
			non_sequences=Qsof_Sp)

		targets = Rs[:-1] + Qsof_Sp_Ap

		Qsof_S_A, _ = scan(lambda i, A, P: P[i,A], \
			sequences=[T.arange(As.shape[0]-1), As[:-1]], \
			non_sequences=Qsof_S)

		cost, updates = _backup(Qsof_S_A, targets, params)

		#shared variables.
		self.Ss_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)
		self.As_shared = shared(np.zeros((0,), dtype='int32'))						# (H, )
		self.Rs_shared = shared(np.zeros((0,), dtype='float32'))					# (H, )
		self.Sps_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)

		self.train_core = function([index], [cost],
			updates=updates,
			givens=[
				(Ss, self.Ss_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(As, self.As_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(Rs, self.Rs_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(Sps, self.Sps_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size])
			])

		def learn():
			Ss = np.array([e[0] for e in self.trajectory], dtype='float32')
			As = np.array([e[1] for e in self.trajectory], dtype='int32')
			Rs = np.array([e[2] for e in self.trajectory], dtype='float32')
			Sps = np.array([e[3] for e in self.trajectory], dtype='float32')
			H = len(Rs)

			self.Ss_shared.set_value(Ss)
			self.As_shared.set_value(As)
			self.Rs_shared.set_value(Rs)
			self.Sps_shared.set_value(Sps)
			
			costs = []
			for i in range(-(-H//self.machine.batch_size)):
				costs.append(self.train_core(i))
			return np.mean(costs)

		self.learn=learn
	
	def build_trainer_n_sarsa(self):
		"""
		:meth:`build_trainer_n_sarsa` compiles a :attr:`train_core` theano function which implements \
		**n-step SARSA**, and algorithm-specific :attr:`learn` function.
		"""
		self.settings['refresh_traj'] = True
		#Since SARSA is an on-policy learning algorithm, trajectories from former pi cannot be used.

		params = self.machine.params
		index = T.iscalar('index')
		Ss = self.machine.S			# (H, 2, h, w)
		As = T.ivector('As')		# (H,)
		Rs = T.vector('Rs')			# (H,)
		Qsof_S = self.machine.P
		Sps = Input(shape=(2, Board.height, Board.width)) # (H, 2, h, w)
		Qsof_Sp = self._get_another_output(Sps, self.machine.model)

		#update formula
		#Vsof_Sp = T.max(Qsof_Sp_Ap, axis=-1)
		Qsof_Sp_Ap, _ = scan(lambda t, A, P: P[t,A], \
			sequences=[T.arange(As.shape[0]-1), As[1:]], \
			non_sequences=Qsof_Sp)

		targets = Rs[:-1] + Qsof_Sp_Ap

		Qsof_S_A, _ = scan(lambda i, A, P: P[i,A], \
			sequences=[T.arange(As.shape[0]-1), As[:-1]], \
			non_sequences=Qsof_S)

		cost, updates = _backup(Qsof_S_A, targets, params)

		#shared variables.
		self.Ss_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)
		self.As_shared = shared(np.zeros((0,), dtype='int32'))						# (H, )
		self.Rs_shared = shared(np.zeros((0,), dtype='float32'))					# (H, )
		self.Sps_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)

		self.train_core = function([index], [cost],
			updates=updates,
			givens=[
				(Ss, self.Ss_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(As, self.As_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(Rs, self.Rs_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(Sps, self.Sps_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size])
			])

		def learn():
			Ss = np.array([e[0] for e in self.trajectory], dtype='float32')
			As = np.array([e[1] for e in self.trajectory], dtype='int32')
			Rs = np.array([e[2] for e in self.trajectory], dtype='float32')
			Sps = np.array([e[3] for e in self.trajectory], dtype='float32')
			H = len(Rs)

			self.Ss_shared.set_value(Ss)
			self.As_shared.set_value(As)
			self.Rs_shared.set_value(Rs)
			self.Sps_shared.set_value(Sps)
			
			costs = []
			for i in range(-(-H//self.machine.batch_size)):
				costs.append(self.train_core(i))
			return np.mean(costs)

		self.learn=learn
		
	def build_trainer_expectedsarsa(self):
		"""
		:meth:`build_trainer_expectedsarsa` compiles a :attr:`train_core` theano function which implements \
		**Expected-SARSA**, and algorithm-specific :attr:`learn` function.
		"""
		self.settings['refresh_traj'] = True
		#Since SARSA is an on-policy learning algorithm, trajectories from former pi cannot be used.

		params = self.machine.params
		index = T.iscalar('index')
		Ss = self.machine.S			# (H, 2, h, w)
		As = T.ivector('As')		# (H,)
		Rs = T.vector('Rs')			# (H,)
		Qsof_S = self.machine.P
		Sps = Input(shape=(2, Board.height, Board.width)) # (H, 2, h, w)

		Qsof_Sp = self._get_another_output(Sps, self.machine.model)		
		Piof_Sp = self.machine.get_pi_from_v(Qsof_Sp)
		ExpectedQsof_Sp = Piof_Sp * Qsof_Sp

		targets = Rs[:-1] + T.sum(ExpectedQsof_Sp[:-1], axis=-1)

		Qsof_S_A, _ = scan(lambda i, A, P: P[i,A], \
			sequences=[T.arange(As.shape[0]-1), As[:-1]], \
			non_sequences=Qsof_S)
		
		cost, updates = self._backup(Qsof_S_A, targets, params)

		#shared variables.
		self.Ss_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)
		self.As_shared = shared(np.zeros((0,), dtype='int32'))						# (H, )
		self.Rs_shared = shared(np.zeros((0,), dtype='float32'))					# (H, )
		self.Sps_shared = shared(np.zeros((0,0,0,0), dtype='float32'))				# (H, 2, h, w)

		self.train_core = function([index], [cost],
			updates=updates,
			givens=[
				(Ss, self.Ss_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(As, self.As_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(Rs, self.Rs_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size]),
				(Sps, self.Sps_shared[index * self.machine.batch_size : (index+1)*self.machine.batch_size])
			])

		def learn():
			Ss = np.array([e[0] for e in self.trajectory], dtype='float32')
			As = np.array([e[1] for e in self.trajectory], dtype='int32')
			Rs = np.array([e[2] for e in self.trajectory], dtype='float32')
			Sps = np.array([e[3] for e in self.trajectory], dtype='float32')
			H = len(Rs)

			self.Ss_shared.set_value(Ss)
			self.As_shared.set_value(As)
			self.Rs_shared.set_value(Rs)
			self.Sps_shared.set_value(Sps)
			
			costs = []
			for i in range(-(-H//self.machine.batch_size)):
				costs.append(self.train_core(i))
			return np.mean(costs)

		self.learn=learn

	def build_trainer_doubledqn(self):
		#READ the paper first..!!
		pass

	def build_trainer_doubleqlearning(self):
		"""
		:meth:`build_trainer_doubleqlearning` compiles a :attr:`train_core` theano function which implements \
		**Double Q-Learning**, and algorithm-specific :attr:`learn` function.
		"""
		pass
		
	##############################################
			
	def start(self):
		"""
		:meth:`start` runs one episode and produces one trajectory.

		After each episode is done, it calls :attr:`learn` function to learn.

		:returns (G, H): (Return, Horizon, cost) cost is a cost estimate for value approximation.
		:rtype: (float, int, cost)
		"""

		self.initialize()
		self.isStarted=True

		startTime = time.time()
		while True:
			self.tick(None)
			if self.isOver:
				break

		cost = self.learn()
		endTime=time.time()
		return self.G, self.board.t, cost

	def train(self):
		"""
		:meth:`train` is a top-level train function. \
		Repeatedly calls :meth:`start` to produce trajectories and learn.

		Every 100 episode, it saves history and model weights.
		"""
		historyname = '%s_%s_%ix%i.pkl'% \
			(self.settings['name'], self.machine.name, self.board.height, self.board.width)

		try:
			with open(historyname, 'rb') as f:
				history = _pickle.load(f)
		except:
			# G for return, and H for horizon
			history = {'G' : [], 'H' : [], 'loss' : []}

		self.Gs = history['G']
		self.Hs = history['H']
		self.Costs = history['loss']

		episode_num = len(history['G'])
		print('Start Value-based learning.')
		firsttime = time.time()
		starttime = firsttime
		Gs_100, Hs_100, costs_100 = [], [], []
		try:
			while True:
				G, H, cost = self.start()
				self.Gs.append(G)
				self.Hs.append(H)
				self.Costs.append(cost)

				Gs_100.append(G)
				Hs_100.append(H)
				costs_100.append(cost)

				episode_num += 1

				if episode_num % 100 == 0:
					nowtime = time.time()
					print('Episode #%i. Mean G : %.2f(%.2f/%.2f), Mean ticks : %.2f(%i/%i). Cost : %.5f. Took %.1fs, elapsed %.1fs'\
						%(episode_num, \
						np.mean(Gs_100), np.min(Gs_100), np.max(Gs_100), \
						np.mean(Hs_100), np.min(Hs_100), np.max(Hs_100), \
						np.mean(costs_100), nowtime-starttime, nowtime-firsttime))
					self.machine.save_weights()
					starttime = nowtime
					with open(historyname, 'wb') as f:
						_pickle.dump(history, f)

					Gs_100, Hs_100, costs_100 = [], [], []
		except KeyboardInterrupt:
			with open(historyname, 'wb') as f:
				_pickle.dump(history, f)