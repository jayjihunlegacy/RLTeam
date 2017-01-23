import numpy as np
import _pickle
import copy
from multiprocessing import Pool, Manager, Process
import os
from functools import reduce
from abc import ABCMeta, abstractmethod
import queue

from ..utils import *
from ..Board import *
from ..Interfaces.BoardInterface import *



class Machine:
	"""
	:class:`Machine` is an abstract base class for Tetris agents.

	:meth:`Pi` must be implemented.
	"""
	__metaclass__ = ABCMeta
	minf = -float('inf')
	maxf = float('inf')
	def __init__(self, proc_num=None):
		self.proc_num = proc_num
		"""Number of processes to use when MC algorithm runs"""
		if self.proc_num is None:
			self.proc_num = os.cpu_count()

		self.workers = None
		"""List of :class:`multiprocessing.Process` instances which are workers"""

		self.Queues = None
		"""List of :class:`queue.Queue` to order workers"""

		self.resQ = None
		"""A :class:`queue.Queue` instance to get results. All workers put results to here."""

		self.width = Board.width
		self.height = Board.height

	def spawn_workers(self):
		"""
		:meth:`spawn_workers` spawns workers if no worker exists, and do nothing if workers exist.

		Use this method in advance any MC algorithm runs.
		"""
		if self.workers is not None:
			return
		self.copies = [copy.deepcopy(self) for i in range(self.proc_num)]
		manager = Manager()
		self.Queues = [manager.Queue() for i in range(self.proc_num)]
		self.resQ = manager.Queue()
		self.workers = [Process(target=self.worker, args=(self.Queues[i], self.resQ, self.copies[i])) \
			for i in range(self.proc_num)]

		for worker in self.workers:
			worker.start()

	def kill_workers(self):
		if self.workers is not None:
			for Queue in self.Queues:
				Queue.put(None)

			for worker in self.workers:
				worker.join()
				
			self.workers = None
			self.Queues = None
			self.resQ = None
		
	def sequential_evaluate(self, N=1000):
		"""
		:meth:`sequential_evaluate` evaluates expected return of start state given policy.\
		This method does not use multiprocessing.

		Simulate a game with *N* many games and return mean and variance of returns.

		:param int N: number of roll-outs

		:returns result: (mean(return), var(return))
		:rtype: tuple
		"""
		s = time.time()

		Rs = []
		dummyInterface = BoardInterface(machine=self)
		for i in range(N):
			r = dummyInterface.start()
			Rs.append(r)

		mean, var = np.mean(Rs), np.var(Rs)

		e = time.time()
		print('Evaluated. Time elapsed : %.1fs'%(e-s))
		return mean, var
	
	@abstractmethod
	def Pi(self, state):
		"""
		:meth:`Pi` returns a probability distribution of actions given *state*.

		:param tuple state: board state in a tuple form. \
		See :meth:`Tetris.mods.Board.Board.S` for details.

		:returns pi: probability distribution of actions. must sum to 1.
		:rtype: list
		"""
		raise NotImplementedError()
		
	def evaluate(self, N=10000):
		"""
		:meth:`evaluate` evaluates expected return of start state given policy.

		Simulate a game with *N* many games and return mean and variance of returns.

		:param int N: number of roll-outs

		:returns result: (mean(return), var(return))
		:rtype: tuple
		"""
		self.spawn_workers()

		s = time.time()
		if N<=100:		
			a = DummyInterface()
			state = a.S()
			mean, var = self.MC_V(state, N=N)
		else:
			a = DummyInterface()
			Means = []
			Vars = []
			for i in range(100):
				a.S0()
				state = a.S()
				mean, var = self.MC_V(state, N=N//10)
				Means.append(mean)
				Vars.append(var)

			mean, var = np.mean(Means), np.mean(Vars)

		e = time.time()
		
		print('Evaluated. Time elapsed : %.1fs'%(e-s))
		self.kill_workers()
		return mean, var

	def MC_V(self, state, maxtime=None, N=None):
		"""
		:meth:`MC_V` calculates Value of a state using Monte Carlo method. \
		Stopping condition of Monte Carlo simulation can be given in two ways: *maxtime* and *N*.

		:param float maxtime: maximum computation time
		:param int N: number of roll-outs

		:returns result: (mean(return), var(return))
		:rtype: tuple
		"""
		assert maxtime is not None or N is not None, "Stopping condition must be specified."
		self.spawn_workers()
		starttime = time.time()

		#1. Empty current resQ
		while True:
			try:
				self.resQ.get(False)
			except queue.Empty:
				break

		#2. Order to workers.
		for Queue in self.Queues:
			Queue.put((1, state))

		returnhistory = []

		#3. Wait for workers.
		for i in range(N):
			res = self.resQ.get()
			returnhistory.append(res)

			if maxtime is not None:
				if time.time() - starttime >= maxtime:
					break

		#4. Stop workers.
		for Queue in self.Queues:
			Queue.put(0)

		return np.mean(returnhistory), np.var(returnhistory)

	def MC_Q(self, state, action, maxtime=None, N=None):
		"""
		:meth:`MC_Q` calculates Value of a state-action using Monte Carlo method. \
		Stopping condition of Monte Carlo simulation can be given in two ways: *maxtime* and *N*.

		:param float maxtime: maximum computation time
		:param int N: number of roll-outs

		:returns result: (mean(return), var(return))
		:rtype: tuple
		"""
		assert maxtime is not None or N is not None, "Stopping condition must be specified."
		self.spawn_workers()
		starttime = time.time()

		#1. Empty current resQ
		while True:
			try:
				self.resQ.get(False)
			except queue.Empty:
				break

		#2. Order to workers.
		for Queue in self.Queues:
			Queue.put((2, state, action))

		returnhistory = []

		#3. Wait for workers.
		for i in range(N):
			res = self.resQ.get()
			returnhistory.append(res)

			if maxtime is not None:
				if time.time() - starttime >= maxtime:
					break

		#4. Stop workers.
		for Queue in self.Queues:
			Queue.put(0)

		return np.mean(returnhistory), np.var(returnhistory)
		
	def worker(self, Q, resQ, machine):
		"""
		:meth:`worker` is a method that executed MC algorithms. \
		It takes a request from Q, and run it.

		Possible requests are:

		1. from :meth:`MC_V`
		2. from :meth:`MC_Q`

		According to input type and shape, it executes following:
		
		1. If None, terminate.
		2. If 0, stop any MC algorithm.
		
		"""
		np.random.seed(int(time.time())+ os.getpid())

		dummyInterface = DummyInterface(machine=machine)
		dummyboard = dummyInterface.board

		try:
			while True:
				req = Q.get()

				#1. If None, terminate.
				if req is None:
					return

				#If some request, perform it.
				if isinstance(req, tuple):

					#########################################
					#3. If (0, state), do MicroMC ##
					if req[0] == 0:
						state = req[1]

						returnhistory = np.zeros(5).astype('int32')
						runhistory = np.zeros(5).astype('int32')

						terminate_now = False

						while not terminate_now:
							dummyInterface.initialize()
							dummyInterface.setS(state)
							A0 = None
							while True:
								S_t = dummyInterface.S()
								A_t = np.random.choice(5, p=self.default_Pi(S_t))
								dummyInterface.perform_action(A_t)
								
								if A0 is None:
									A0 = A_t

								if not Q.empty():
									try:
										req = Q.get(False)
										#2. If 0, stop any MC algorithm.
										if req==0:
											terminate_now=True
											break
										elif req is None:
											return
									except EOFError or BrokenPipeError:
										return
								
								if dummyboard.isOver:
									break

							returnhistory[A0] += dummyInterface.G
							runhistory[A0] += 1

						resQ.put((returnhistory.copy(), runhistory.copy()))

					#############################################
					#4. If (1, state), do MC until termination ##
					elif req[0] == 1:
						state = req[1]
						while True:
							dummyInterface.initialize()
							dummyInterface.isStarted=True
							dummyInterface.setS(state)						
							while True:
								dummyInterface.tick(None)
								if dummyInterface.isOver:
									break

								try:
									req = Q.get(False)
									#2. If 0, stop any MC algorithm.
									if req==0:
										break
									elif req is None:
										return
								except queue.Empty:
									pass

							resQ.put(dummyInterface.G)

					#####################################################
					#5. If (2, state, action), do MC until termination ##
					elif req[0] == 2:
						state = req[1]
						action = req[2]
					
						while True:
							dummyInterface.initialize()
							dummyInterface.setS(state)
						
							dummyInterface.perform_action(action)
							while True:
								dummyInterface.tick(None)
								if dummyInterface.isOver:
									break

							resQ.put(dummyInterface.G)

							try:
								req = Q.get(False)
								#2. If 0, stop any MC algorithm.
								if req==0:
									break
								elif req is None:
									return
							except queue.Empty:
								pass
		except KeyboardInterrupt:
			pass
		
	def _evaluate_worker(self, tup):
		machine, n = tup
		results = []
		interface = BoardInterface(machine=machine)
		for i in range(n):
			r = interface.start()
			results.append(r)
		return results
		
	def phi(self, state):
		"""
		:meth:`phi` encodes state tuple into (2, 22, 10) tensor.
		
		:param tuple state: board state in a tuple form. \
		See :meth:`Tetris.mods.Board.Board.S` for details.
		
		:return phi: binary (2, 22, 10) state tensor
		:rtype: numpy.ndarray
		"""
		board, pieces, piececoords, curX, curY, ticks, isOver = state
		
		x_piece = np.zeros((Board.height, Board.width), dtype='int32')
		Xs = curX + piececoords[:, 0]
		Ys = curY - piececoords[:, 1]
		x_piece[Ys,Xs]=1		
		X = np.stack([(board > 0).astype('int32'), x_piece])

		return X

	def heuristic_V(self, S):
		"""
		:meth:`heuristic_V` evaluates Value of a state heuristically.

		:param S: binary (2, 22, 10) state tensor
		:type S: numpy.ndarray

		:return V: heuristic value
		:rtype: int or float
		"""
		#1. number of holes.
		hole_num = 0
		upperExist = np.zeros(Board.width).astype('bool')
		for row in S[::-1]:
			upperExist = np.logical_or(row, upperExist)
			hole_num += np.sum(np.logical_and(1-row, upperExist))

		#2. penalty sum.
		penalty_sum = np.sum( np.dot(self.penalty_column, S) )

		#return.
		x = [hole_num, penalty_sum]
		result = np.dot(x, self.w)
		return result
	
class StochasticMachine(Machine):
	def __init__(self, ep, cool_time=1):
		super().__init__()
		self.TICK_COOLTIME = cool_time
		self.ep = ep

		self.dummyInterface = DummyInterface()
		self.dummyboard=self.dummyInterface.board
				
		self.initialize()

		self.TT = []
		self.CONSIDERS = []

	def initialize(self):
		#1. number of holes.
		#2. height penalty sum.
		self.w = np.array([-10, -1])
		self.penalty_column = np.square([np.arange(Board.height)])

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
		self.mA2uA=dict()
		
		for shape in shapes:
			self.mAs[shape] = list()
			for rotate in range(effect_rots[shape]):
				self.mAs[shape] += [[left, rotate] for left in range(-(Board.width-1),(Board.width-1))]

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

		self.mA = None

	def Pi(self, state):
		board, pieces, piececoords, curX, curY, ticks, isOver = state
		curshape = pieces[0]

		use_random = np.random.binomial(1, self.ep, 1)[0]
		if use_random:
			return [0.2, 0.2, 0.2, 0.2, 0.2]
		
		#1. When T-shape on top, move one below!
		if curshape == Tetrominoes.TShape and curY==Board.height-1:
			return [0, 0, 0, 1, 0]

		#2. When aimPosition is None, set it.
		if self.mA is None:			
			con = self._setmA(state)
		#3. give output.
		if ticks % self.TICK_COOLTIME == 0:
			action = self._getuA(state)
			pi = [0, 0, 0, 0, 0]
			pi[action]=1
			return pi

		else:
			return None

	def _getuA(self, state):
		# if have to rotate
		if self.mA[1]:
			self.mA[1]-=1
			return Action.UP
			
		# if have to go left:
		if self.mA[0] > 0:
			self.mA[0]-=1
			return Action.LEFT
			
		#if have to go right:
		if self.mA[0] < 0:
			self.mA[0]+=1
			return Action.RIGHT
			
		# nothing to do any more.
		self.mA = None
		return Action.SPACE

	def _setmA(self, state):
		board, pieces, piececoords, curX, curY, ticks, isOver = state
		curshape = pieces[0]
		Qs = [self.Q(state, uAseq) for uAseq in self.mA2uA[curshape]]
		mA_idx = np.argmax(Qs)		
		self.mA = list(self.mAs[curshape][mA_idx])

	def Q(self, state, uAseq):
		self.dummyInterface.setS(state)
		r = 0
		for uA in uAseq:
			r_, wasValid = self.dummyInterface.perform_action(uA)
			r += r_
			if not wasValid:
				return Machine.minf
		else:
			r_, _ =  self.dummyInterface.perform_action(Action.SPACE)
			r += r_
			Sprime = (self.dummyboard.table > 0).astype('int32')
			return self.heuristic_V(Sprime) + r

class StochasticMachine2(Machine):
	def __init__(self, ep, cool_time=1):
		super().__init__()
		self.TICK_COOLTIME = cool_time
		self.ep = ep

		self.dummyInterface = DummyInterface()
		self.dummyboard=self.dummyInterface.board
				
		self.initialize()

		self.TT = []
		self.CONSIDERS = []

	def initialize(self):
		#1. number of holes.
		#2. height penalty sum.
		self.w = np.array([-60, -1])

		self.penalty_column = np.square([np.arange(Board.height)])

		shapes = [Tetrominoes.ZShape, 
			Tetrominoes.SShape, 
			Tetrominoes.LineShape, 
			Tetrominoes.TShape, 
			Tetrominoes.SquareShape, 
			Tetrominoes.LShape, 
			Tetrominoes.MirroredLShape]	

		self.eff_rots={
			Tetrominoes.ZShape : 2,
			Tetrominoes.SShape : 2,
			Tetrominoes.LineShape : 2,
			Tetrominoes.TShape : 4,
			Tetrominoes.SquareShape : 1,
			Tetrominoes.LShape : 4,
			Tetrominoes.MirroredLShape : 4,
			}

		self.mAroot = Node(3) # left, right, up
		ptr_rotate = self.mAroot

		#build computation tree
		for rotate in range(4):			
			ptr = ptr_rotate
			for left in range(self.dummyInterface.width-1):
				ptr = ptr.get_child(Action.LEFT)

			ptr = ptr_rotate
			for right in range(self.dummyInterface.width-1):
				ptr = ptr.get_child(Action.RIGHT)

			if rotate != 3:
				ptr_rotate = ptr_rotate.get_child(Action.UP, 1)
		
		self.uAseq = None

	def Pi(self, state):
		board, pieces, piececoords, curX, curY, ticks, isOver = state
		curshape = pieces[0]

		if np.random.binomial(1, self.ep, 1)[0]:
			return [0.2, 0.2, 0.2, 0.2, 0.2]
		
		#1. When T-shape on top, move one below!
		if curshape == Tetrominoes.TShape and curY==Board.height-1:
			return [0, 0, 0, 1, 0]

		#2. When aimPosition is None, set it.
		if self.uAseq is None:			
			self._setmA(state)

		#3. give output.
		if ticks % self.TICK_COOLTIME == 0:
			action = self._getuA(state)
			pi = [0, 0, 0, 0, 0]
			pi[action]=1
			return pi

		else:
			return None

	def _setmA(self, state):
		#set self.uAseq
		board, pieces, piececoords, curX, curY, ticks, isOver = state
		curshape = pieces[0]
		eff_rot = self.eff_rots[curshape]
		self.dummyInterface.setS(state)
		
		Q, uAseq = self.Q(self.mAroot, eff_rot)
		self.uAseq = uAseq

	
	def _getuA(self, state):
		if len(self.uAseq) != 0:
			uA = self.uAseq[0]
			self.uAseq = self.uAseq[1:]
		else:
			uA = Action.SPACE
			self.uAseq = None
		return uA

	def Q(self, node, eff_rot):			
		#return Q, [uAseq]
		ptr = node			

		#1. save state
		before = self.dummyInterface.S()

		#2. drop piece
		r, _ =  self.dummyInterface.perform_action(Action.SPACE)
		Sprime = (self.dummyboard.table > 0).astype('int32')
		V = self.heuristic_V(Sprime) + r

		#3. backup
		self.dummyInterface.setS(before)

		Qs = [self.minf] * 5
		uAseqs = [None] * 5


		#4. move to child
		for mA in node.avail_actions(eff_rot):
			r, wasValid = self.dummyInterface.perform_action(mA)
			if wasValid:
				ptr = node.mov_child(mA)					
				Qsum2, uAseq2 = self.Q(ptr, eff_rot)
				Qs[mA] = Qsum2 + r
				uAseqs[mA] = uAseq2

			self.dummyInterface.setS(before)
		
		max_mA = np.argmax(Qs)
			
		if Qs[max_mA] <= V:
			return V, []
		else:
			return Qs[max_mA], [max_mA] + uAseqs[max_mA]

