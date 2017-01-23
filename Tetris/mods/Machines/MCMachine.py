import queue
import numpy as np
from abc import ABCMeta, abstractmethod
import _pickle
import copy
import os
from functools import reduce
from multiprocessing import Process, Manager

from .Machine import Machine

from ..Board import *
from ..Interfaces.BoardInterface import *
from ..utils import *


class MicroMCMachine(Machine):
	"""
	:class:`MicroMCMachine` is an Monte Carlo machine that samples Q from micro actions.

	This machine has a Markov Property. i.e. can be deployed to random state with no history.

	:param float maxtime: maximum computation time
	:param int proc_num: number of processes to use when MC algorithm runs
	"""
	def __init__(self, maxtime=.1, proc_num=None):
		super().__init__(proc_num)

		self.maxtime = maxtime
		"""Maximum computation time"""
		
		self.verbose_mc = True
		
		self.initialize()

	evaluate=Machine.sequential_evaluate

	def __del__(self):		
		if self.workers is not None:
			for Queue in self.Queues:
				try:
					Queue.put(None)
				except:
					pass

			for worker in self.workers:			
				try:
					worker.join()
				except:
					pass

	def initialize(self):
		"""
		:meth:`initialize` spawns :attr:`proc_num` many workers and starts them.
		"""
		self.spawn_workers()

	def default_Pi(self, state):
		"""
		:meth:`default_Pi` is a policy used during MC simulation.

		:param tuple state: board state in a tuple form. \
		See :meth:`Tetris.mods.Board.Board.S` for details.

		:returns pi: probability distribution of actions. must sum to 1.
		:rtype: list
		"""
		return [0.2, 0.2, 0.1, 0.05, 0.45]

	def Pi(self, state):
		"""
		:meth:`Pi` returns a probability distribution of actions given *state*. \
		This method does not run any time-consuming code. Instead, it directs workers to run \
		simulations, gathers data, and returns maximal action.		

		:param tuple state: board state in a tuple form. \
		See :meth:`Tetris.mods.Board.Board.S` for details.

		:returns pi: probability distribution of actions. must sum to 1.
		:rtype: list
		"""
		starttime = time.time()
		self.spawn_workers()

		#1. Empty current resQ
		while True:
			try:
				self.resQ.get(False)
			except queue.Empty:
				break

		#2. Order to workers.
		for Queue in self.Queues:
			Queue.put((0, state))

		#3. Wait for workers.
		while True:
			time.sleep(self.maxtime/10)
			if time.time() - starttime >= self.maxtime:
				break

		#4. Stop workers.
		for Queue in self.Queues:
			Queue.put(0)

		RperAs = []
		runhistories = []

		#5. Gather data from workers.
		for i in range(self.proc_num):
			try:
				RperA, runhistory = self.resQ.get(timeout=1)
				RperAs.append(RperA)
				runhistories.append(runhistory)
			except queue.Empty:
				print('Timeout 10 seconds..')

		RperA = np.sum(RperAs, axis=0)
		runhistory = np.sum(runhistories, axis=0)
		#WHAT IF DEFAULT POLICY IS TOO GOOD AND NO RETURN IS AVAILABLE???
		try:
			runhistory[runhistory==0] = 1
		except TypeError:
			print('runhistory :',runhistory)
			print('runhistories :',runhistories)

		RperA = np.divide(RperA, runhistory)

		if self.verbose_mc:
			print('Iter : ', np.sum(runhistory), str(np.sum(runhistories,axis=1)), '/ R per Actions:',RperA)

		if np.sum(RperA) == 0:
			return None
		else:
			result = [0,0,0,0,0]		
			result[np.argmax(RperA)] = 1
			return result
		
class MacroMCMachine(Machine):
	def __init__(self, maxtime=1, cool_time=1):
		super().__init__()
		self.TICK_COOLTIME = cool_time
		self.maxtime = maxtime

		self.dummyInterface = DummyInterface()
		self.dummyboard = self.dummyInterface.board

		self.initialize()
		
	def initialize(self):
		shapes = [Tetrominoes.ZShape, 
			Tetrominoes.SShape, 
			Tetrominoes.LineShape, 
			Tetrominoes.TShape, 
			Tetrominoes.SquareShape, 
			Tetrominoes.LShape, 
			Tetrominoes.MirroredLShape]
			
		effect_rots={
			Tetrominoes.ZShape : 2,
			Tetrominoes.SShape : 2,
			Tetrominoes.LineShape : 2,
			Tetrominoes.TShape : 4,
			Tetrominoes.SquareShape : 1,
			Tetrominoes.LShape : 4,
			Tetrominoes.MirroredLShape : 4,
			}

		self.mAs=dict()
		for shape in shapes:
			self.mAs[shape]=list()
			for rotate in range(effect_rots[shape]):
				self.mAs[shape] += [[left, rotate] \
					for left in range(-(self.dummyInterface.width-1),(self.dummyInterface.width-1))]
		
		self.mA2uA=dict()
		for shape in shapes:
			self.mA2uA[shape]=list()
			for mA in self.mAs[shape]:
				uAseq = list()
				if mA[1]:
					uAseq+=[Action.UP for i in range(mA[1])]

				if mA[0]>0:
					uAseq+=[Action.LEFT for i in range(mA[0])]
				else:
					uAseq+=[Action.RIGHT for i in range(-mA[0])]
				self.mA2uA[shape].append(uAseq)

		self.root = StateActionNode(np.max([len(self.mAs[shape]) for shape in self.mAs]))
		self.mA = None

	def _getuA(self, state):		
		if self.mA[1]:
			self.mA[1]-=1
			return Action.UP # if have to rotate
					
		if self.mA[0] > 0:
			self.mA[0]-=1
			return Action.LEFT # if have to go left
					
		if self.mA[0] < 0:
			self.mA[0]+=1
			return Action.RIGHT #if have to go right
					
		self.mA = None
		return Action.SPACE # nothing to do any more.

	def Pi(self, state):
		board, pieces, piececoords, curX, curY, ticks, isOver = state
		curshape = pieces[0]		
		
		if curshape == Tetrominoes.TShape and curY==Board.height-1:
			return [0, 0, 0, 1, 0]#When T-shape on top, move one below.

		if self.mA is None:			
			self._setmA(state)#When aimPosition is None, set it.
			
		if ticks % self.TICK_COOLTIME == 0:
			action = self._getuA(state)
			pi = [0, 0, 0, 0, 0]
			pi[action] = 1
			return pi

		else:
			return None

	def _setmA(self, state):
		board, pieces, piececoords, curX, curY, ticks, isOver = state
		curshape = pieces[0]
		starttime = time.time()
		maxstep = 5
		iter = 0
		iter_suc=0
		#self.root.clear()
		while True:
			iter+=1
			self.dummyInterface.initialize()
			self.dummyInterface.setS(state)
			pointer = self.root

			G=0
			for t in range(maxstep):
				before = self.dummyInterface.S()
				simshape = before[1][0]
				num_mA = len(self.mAs[simshape])
				mA_idx = np.random.randint(num_mA) #choose macro Action

				uAseq = self.mA2uA[simshape][mA_idx]

				for uA in uAseq:
					r, wasValid =  self.dummyInterface.perform_action(uA)
					if not wasValid:
						self.dummyInterface.setS(before)
						break
					G+=r
					if r!=0:
						print(r)
				else:
					pointer = pointer.get_child(mA_idx)				
			else:
				pointer.backprop(G)			
				iter_suc+=1

			if time.time() - starttime >= self.maxtime:
				break
					
		Qs = self.root.get_Qs()
		mA_idx = np.argmax(Qs)
		self.root = self.root.get_child(mA_idx)
		self.root.parent=None
		print('Iter : %i, Iter_suc : %i, Qs : %s'%(iter, iter_suc, str(Qs)))
		
		
		self.mA = list(self.mAs[curshape][mA_idx])
			
	def _worker(self, Q, resQ):
		np.random.seed(int(time.time() + os.getpid()))
		dummyInterface = DummyInterface()
		dummyboard = self.dummyInterface.board
		maxstep = 5
		returnhistory = np.zeros(5).astype('int32')
		runhistory = np.zeros(5).astype('int32')
		
		while True:
			req = Q.get()
			if req is None:
				return

			if isinstance(req, tuple):
				state = req

				returnhistory[:] = 0
				runhistory[:] = 0

				while True:
					dummyInterface.initialize()
					dummyInterface.setS(state)

					pointer = self.root
					G = 0
					for t in range(maxstep):
						before = dummyInterface.S()
						simshape = before[1][0]
						mA_idx = np.random.choice(len(self.mAs[simshape]))
						uAseq = self.mA2uA[simshape][mA_idx]
						for uA in uAseq:
							r, wasValid = dummyInterface.perform_action(uA)
							if not wasValid:
								dummyInterface.setS(before)
								break
							G+=r
							if r!=0:
								print(os.getpid(), r)

						else:
							pointer = pointer.get_child(mA_idx)

					else:
						pointer.backprop(G)

					try:
						a = Q.get(False)
						
						if a==0:
							break
						elif isinstance(a, tuple):
							state = a
							returnhistory[:] = 0
							runhistory[:] = 0

					except queue.Empty:
						pass
				resQ.put((returnhistory.copy(), runhistory.copy()))
				
class SteppedMicroMCMachine(Machine):
	"""
	:class:`MicroMCMachine` is an Monte Carlo machine that samples Q from micro actions.

	This machine has a Markov Property. i.e. can be deployed to random state with no history.

	:param float maxtime: maximum computation time
	:param int proc_num: number of processes to use when MC algorithm runs
	"""
	def __init__(self, maxstep=20, maxtime=.2, proc_num=None):
		super().__init__(proc_num)

		self.maxtime = maxtime
		"""Maximum computation time"""
		
		self.maxstep = maxstep
		"""Maximum step to look ahead"""

		self.verbose_mc = False
		
		self.initialize()

	evaluate=Machine.sequential_evaluate

	def __del__(self):		
		if self.Queues is not None:
			for Queue in self.Queues:
				try:
					Queue.put(None)
				except:
					pass

			for worker in self.workers:			
				worker.join()

	def initialize(self):
		"""
		:meth:`initialize` spawns :attr:`proc_num` many workers and starts them.
		"""
		self.spawn_workers()

	def default_Pi(self, state):
		"""
		:meth:`default_Pi` is a policy used during MC simulation.

		:param tuple state: board state in a tuple form. \
		See :meth:`Tetris.mods.Board.Board.S` for details.

		:returns pi: probability distribution of actions. must sum to 1.
		:rtype: list
		"""
		return [0.2, 0.2, 0.1, 0.05, 0.45]

	def Pi(self, state):
		"""
		:meth:`Pi` returns a probability distribution of actions given *state*. \
		This method does not run any time-consuming code. Instead, it directs workers to run \
		simulations, gathers data, and returns maximal action.		

		:param tuple state: board state in a tuple form. \
		See :meth:`Tetris.mods.Board.Board.S` for details.

		:returns pi: probability distribution of actions. must sum to 1.
		:rtype: list
		"""
		starttime = time.time()
		self.spawn_workers()

		#1. Empty current resQ
		while True:
			try:
				self.resQ.get(False)
			except queue.Empty:
				break

		#2. Order to workers.
		for Queue in self.Queues:
			Queue.put((0, state, self.maxstep))

		#3. Wait for workers.
		while True:
			time.sleep(self.maxtime/10)
			if time.time() - starttime >= self.maxtime:
				break

		#4. Stop workers.
		for Queue in self.Queues:
			Queue.put(0)

		RperAs = []
		runhistories = []

		#5. Gather data from workers.
		while True:
			try:
				RperA, runhistory = self.resQ.get(timeout=0.01)
				RperAs.append(RperA)
				runhistories.append(runhistory)
			except queue.Empty:
				break

		RperA = np.sum(RperAs, axis=0)
		runhistory = np.sum(runhistories, axis=0)
		runhistory[runhistory==0] = 1

		RperA = np.divide(RperA, runhistory)

		if self.verbose_mc:
			print('Iter : ', np.sum(runhistory), str(np.sum(runhistories,axis=1)), '/ R per Actions:',RperA)

		if np.sum(RperA) == 0:
			return None
		else:
			result = [0,0,0,0,0]		
			result[np.argmax(RperA)] = 1
			return result

class SteppedMacroMCMachine(Machine):
	def __init__(self, maxtime=1, cool_time=1):
		super().__init__()
		self.TICK_COOLTIME = cool_time
		self.maxtime = maxtime

		self.dummyInterface = DummyInterface()
		self.dummyboard = self.dummyInterface.board

		self.initialize()
		
	def initialize(self):
		shapes = [Tetrominoes.ZShape, 
			Tetrominoes.SShape, 
			Tetrominoes.LineShape, 
			Tetrominoes.TShape, 
			Tetrominoes.SquareShape, 
			Tetrominoes.LShape, 
			Tetrominoes.MirroredLShape]
			
		effect_rots={
			Tetrominoes.ZShape : 2,
			Tetrominoes.SShape : 2,
			Tetrominoes.LineShape : 2,
			Tetrominoes.TShape : 4,
			Tetrominoes.SquareShape : 1,
			Tetrominoes.LShape : 4,
			Tetrominoes.MirroredLShape : 4,
			}

		self.mAs=dict()
		for shape in shapes:
			self.mAs[shape]=list()
			for rotate in range(effect_rots[shape]):
				self.mAs[shape] += [[left, rotate] \
					for left in range(-(self.dummyInterface.width-1),(self.dummyInterface.width-1))]
		
		self.mA2uA=dict()
		for shape in shapes:
			self.mA2uA[shape]=list()
			for mA in self.mAs[shape]:
				uAseq = list()
				if mA[1]:
					uAseq+=[Action.UP for i in range(mA[1])]

				if mA[0]>0:
					uAseq+=[Action.LEFT for i in range(mA[0])]
				else:
					uAseq+=[Action.RIGHT for i in range(-mA[0])]
				self.mA2uA[shape].append(uAseq)

		self.root = StateActionNode(np.max([len(self.mAs[shape]) for shape in self.mAs]))
		self.mA = None

	def _getuA(self, state):		
		if self.mA[1]:
			self.mA[1]-=1
			return Action.UP # if have to rotate
					
		if self.mA[0] > 0:
			self.mA[0]-=1
			return Action.LEFT # if have to go left
					
		if self.mA[0] < 0:
			self.mA[0]+=1
			return Action.RIGHT #if have to go right
					
		self.mA = None
		return Action.SPACE # nothing to do any more.

	def Pi(self, state):
		board, pieces, piececoords, curX, curY, ticks, isOver = state
		curshape = pieces[0]		
		
		if curshape == Tetrominoes.TShape and curY==Board.height-1:
			return [0, 0, 0, 1, 0]#When T-shape on top, move one below.

		if self.mA is None:			
			self._setmA(state)#When aimPosition is None, set it.
			
		if ticks % self.TICK_COOLTIME == 0:
			action = self._getuA(state)
			pi = [0, 0, 0, 0, 0]
			pi[action] = 1
			return pi

		else:
			return None

	def _setmA(self, state):
		board, pieces, piececoords, curX, curY, ticks, isOver = state
		curshape = pieces[0]
		starttime = time.time()
		maxstep = 5
		iter = 0
		iter_suc=0
		#self.root.clear()
		while True:
			iter+=1
			self.dummyInterface.initialize()
			self.dummyInterface.setS(state)
			pointer = self.root

			G=0
			for t in range(maxstep):
				before = self.dummyInterface.S()
				simshape = before[1][0]
				num_mA = len(self.mAs[simshape])
				mA_idx = np.random.randint(num_mA) #choose macro Action

				uAseq = self.mA2uA[simshape][mA_idx]

				for uA in uAseq:
					r, wasValid =  self.dummyInterface.perform_action(uA)
					if not wasValid:
						self.dummyInterface.setS(before)
						break
					G+=r
					if r!=0:
						print(r)
				else:
					pointer = pointer.get_child(mA_idx)				
			else:
				pointer.backprop(G)			
				iter_suc+=1

			if time.time() - starttime >= self.maxtime:
				break
					
		Qs = self.root.get_Qs()
		mA_idx = np.argmax(Qs)
		self.root = self.root.get_child(mA_idx)
		self.root.parent=None
		print('Iter : %i, Iter_suc : %i, Qs : %s'%(iter, iter_suc, str(Qs)))
		
		
		self.mA = list(self.mAs[curshape][mA_idx])
			
	def _worker(self, Q, resQ):
		np.random.seed(int(time.time() + os.getpid()))
		dummyInterface = DummyInterface()
		dummyboard = self.dummyInterface.board
		maxstep = 5
		returnhistory = np.zeros(5).astype('int32')
		runhistory = np.zeros(5).astype('int32')
		
		while True:
			req = Q.get()
			if req is None:
				return

			if isinstance(req, tuple):
				state = req

				returnhistory[:] = 0
				runhistory[:] = 0

				while True:
					dummyInterface.initialize()
					dummyInterface.setS(state)

					pointer = self.root
					G = 0
					for t in range(maxstep):
						before = dummyInterface.S()
						simshape = before[1][0]
						mA_idx = np.random.choice(len(self.mAs[simshape]))
						uAseq = self.mA2uA[simshape][mA_idx]
						for uA in uAseq:
							r, wasValid =  dummyInterface.perform_action(uA)
							if not wasValid:
								dummyInterface.setS(before)
								break
							G+=r
							if r!=0:
								print(os.getpid(), r)

						else:
							pointer = pointer.get_child(mA_idx)

					else:
						pointer.backprop(G)

					try:
						a = Q.get(False)
						
						if a==0:
							break
						elif isinstance(a, tuple):
							state = a
							returnhistory[:] = 0
							runhistory[:] = 0

					except queue.Empty:
						pass
				resQ.put((returnhistory.copy(), runhistory.copy()))

class MCTSMachine(Machine):
	def __init__(self, maxtime=.2):
		super().__init__()
		self.maxtime = maxtime

		self.dummyInterface = DummyInterface()
		self.dummyboard = self.dummyInterface.board

		self.initialize()
		

	def initialize(self):
		self.root = StateActionNode(5) # because micro!

	def Pi(self, state):
		starttime = time.time()
		default_pi = [0.3,0.3,0.1,0.1,0.2]
		maxstep = 20
		iter = 0
		while True:
			iter+=1
			self.dummyInterface.initialize()
			self.dummyInterface.setS(state)
			pointer = self.root

			for t in range(maxstep):
				action = np.random.choice(5, p=default_pi)
				self.dummyInterface.perform_action(action)
				pointer = pointer.get_child(action)

			G = self.dummyInterface.G
			#backpropagation
			while pointer is not None:
				pointer.update(R)
				pointer = pointer.parent
			

			if time.time() - starttime >= self.maxtime:
				break
					
		Qs = self.root.get_Qs()
		self.root = self.root.get_child(action)
		self.root.parent=None
		print('Iter :',iter, 'Qs',Qs)

		try:
			exp_Qs = np.exp(Qs)
			pi = exp_Qs/np.sum(exp_Qs)
		except:
			pi = default_pi

		action = np.random.choice(5, p=pi)
		
		result = [0,0,0,0,0]
		result[action] = 1
		return result
