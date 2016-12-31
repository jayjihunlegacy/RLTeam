import numpy as np
from Board import *
from Interfaces.BoardInterface import *
from abc import ABCMeta, abstractmethod
import _pickle
import copy
from multiprocessing import Pool
import os
from functools import reduce
from utils import *

def one(tup):
	machine, n = tup
	machine = copy.deepcopy(machine)
	results = []
	interface = BoardInterface(machine=machine)
	for i in range(n):
		r = interface.start()
		results.append(r)
	return results

class Machine:
	__metaclass__ = ABCMeta
	minf = -float('inf')
	maxf = float('inf')
	def initialize(self):
		pass

	@abstractmethod
	def Pi(self, state):
		raise NotImplementedError()
			
	def evaluate(self, N=12):
		proc_num = os.cpu_count()
		s = time.time()
		with Pool() as pool:
			results = pool.map(one, [(self, N//proc_num)] * proc_num)
		Rs = reduce(lambda x,y:x+y, results)
		e = time.time()

		print('Evalute completed. Time elapsed : %.1fs'%(e-s))
		return np.mean(Rs), np.var(Rs)
		
	def phi(self, state):
		board, pieces, piececoords, curX, curY, ticks, isOver = state
		
		x_piece = np.zeros((Board.height, Board.width), dtype='int32')
		Xs = curX + piececoords[:, 0]
		Ys = curY - piececoords[:, 1]
		x_piece[Ys,Xs]=1
		
		X = np.stack([(board > 0).astype('int32'), x_piece])		
		return X

	def heuristic_V(self, S):
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
				self.mAs[shape] += [[left, rotate] for left in range(-9,9)]

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
		self.dummyboard.setS(state)
		r = 0
		for uA in uAseq:
			r_, wasValid = self.dummyboard.T(uA)
			r += r_
			if not wasValid:
				return Machine.minf
		else:
			r_, _ = self.dummyboard.T(Action.SPACE)
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
			for left in range(9):
				ptr = ptr.get_child(Action.LEFT)

			ptr = ptr_rotate
			for right in range(9):
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
		self.dummyboard.setS(state)
		
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
		before = self.dummyboard.S()

		#2. drop piece
		r, _ = self.dummyboard.T(Action.SPACE)
		Sprime = (self.dummyboard.table > 0).astype('int32')
		V = self.heuristic_V(Sprime) + r

		#3. backup
		self.dummyboard.setS(before)

		Qs = [self.minf] * 5
		uAseqs = [None] * 5


		#4. move to child
		for mA in node.avail_actions(eff_rot):
			r, wasValid = self.dummyboard.T(mA)
			if wasValid:
				ptr = node.mov_child(mA)					
				Qsum2, uAseq2 = self.Q(ptr, eff_rot)
				Qs[mA] = Qsum2 + r
				uAseqs[mA] = uAseq2

			self.dummyboard.setS(before)
		
		max_mA = np.argmax(Qs)
			
		if Qs[max_mA] <= V:
			return V, []
		else:
			return Qs[max_mA], [max_mA] + uAseqs[max_mA]

