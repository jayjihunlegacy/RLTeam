import numpy as np
from abc import ABCMeta, abstractmethod
import _pickle
import copy
import os
from functools import reduce

from .Machine import Machine

from ..Board import *
from ..Interfaces.BoardInterface import *
from ..utils import *

class MicroMCMachine(Machine):
	def __init__(self, maxtime=10):
		super().__init__()
		self.maxtime = maxtime

		self.dummyInterface = DummyInterface()
		self.dummyboard = self.dummyInterface.board

		self.initialize()

	def initialize(self):
		self.root = StateActionNode(5) # because micro!

	def default_Pi(self, state):
		return [0.2, 0.2, 0.1, 0.05, 0.45]

	def Pi(self, state):
		starttime = time.time()

		maxstep = 30

		returnhistories = {
			Action.LEFT : list(),
			Action.RIGHT : list(),
			Action.UP : list(),
			Action.DOWN : list(),
			Action.SPACE : list()
			}
		iterations=0
		while True:
			iterations+=1
			#print(iterations)
			self.dummyInterface.initialize()
			self.dummyboard.setS(state)
			A = []
			for t in range(maxstep):
				S_t = self.dummyboard.S()
				A_t = np.random.choice(5, p=self.default_Pi(S_t))
				self.dummyInterface.perform_action(A_t)
				A.append(A_t)

				if self.dummyboard.isOver:
					break
						
			returnhistories[A[0]].append(self.dummyInterface.R)

			if time.time() - starttime >= self.maxtime:
				break

		RperA = []
		for a in returnhistories:
			RperA.append(np.mean(returnhistories[a]))

		print('Iter :',iterations, 'R per Actions:',RperA)
		a = np.argmax(RperA)
		if np.sum(RperA) == 0:
			return None
		else:
			result = [0,0,0,0,0]		
			result[a] = 1
		return result

class MacroMCMachine(Machine):
	def __init__(self, maxtime=5, cool_time=1):
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
				self.mAs[shape] += [[left, rotate] for left in range(-9,9)]
		
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
		while True:
			iter+=1
			self.dummyInterface.initialize()
			self.dummyboard.setS(state)
			pointer = self.root

			R=0
			for t in range(maxstep):
				before = self.dummyboard.S()
				simshape = before[1][0].shape()
				mA_idx = np.random.choice(len(self.mAs[simshape])) #choose macro Action
				uAseq = self.mA2uA[simshape][mA_idx]
				for uA in uAseq:
					r, wasValid = self.dummyboard.T(uA)
					if not wasValid:
						self.dummyboard.setS(before)
						break
					R+=r
					if r!=0:
						print(r)
				else:
					pointer = pointer.get_child(mA_idx)				
			else:
				pointer.backprop(R)			

			if time.time() - starttime >= self.maxtime:
				break
					
		Qs = self.root.get_Qs()
		mA_idx = np.argmax(Qs)
		self.root = self.root.get_child(mA_idx)
		self.root.parent=None
		print('Iter :',iter, 'Qs',Qs)
		
		
		self.mA = list(self.mAs[curshape][mA_idx])

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
			self.dummyboard.setS(state)
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
