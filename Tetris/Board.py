
import _pickle
import numpy as np
import time
from abc import ABCMeta, abstractmethod
class Action:
	LEFT = 0
	RIGHT = 1
	UP = 2
	DOWN = 3
	SPACE = 4
	STEP = 5
	tostr={
		LEFT :	'LEFT',
		RIGHT :	'RIGHT',
		UP :	'UP',
		DOWN :	'DOWN',
		SPACE :	'SPACE'
	}

class Reward:
	MOVEMENT = 0
	INVALID_MOVEMENT = 0
	LINECLEAR = 1
	STEP = 0
	GAMEOVER = 0

class Tetrominoes:
	coordsTable = np.array([
		[[0, 0],     [0, 0],     [0, 0],     [0, 0]], #NoShape
        [[0, -1],    [0, 0],     [-1, 0],    [-1, 1]],#ZShape
        [[0, -1],    [0, 0],     [1, 0],     [1, 1]], #SShape
        [[0, -1],    [0, 0],     [0, 1],     [0, 2]], #LineShape
        [[-1, 0],    [0, 0],     [1, 0],     [0, 1]], #TShape
        [[0, 0],     [1, 0],     [0, 1],     [1, 1]], #SquareShape
        [[-1, -1],   [0, -1],    [0, 0],     [0, 1]], #LShape
        [[1, -1],    [0, -1],    [0, 0],     [0, 1]]  #MirroredLShape
		], dtype='int32')

	NoShape = 0
	ZShape = 1
	SShape = 2
	LineShape = 3
	TShape = 4
	SquareShape = 5
	LShape = 6
	MirroredLShape = 7
	tostr={
		NoShape	:	'Noshape',
		ZShape :	'Z-shape',
		SShape :	'S-shape',
		LineShape :	'I-shape',
		TShape :	'T-shape',
		SquareShape:'Square',
		LShape :	'L-shape',
		MirroredLShape:"L'-shape"
		}

	#Added type (Not Shape, for drawing or special block)
	Aim = 8

class MDP:
	__metaclass__ = ABCMeta
	@abstractmethod
	def T(self, action):
		'''
		Make a transition with given action.

		Returns
		----------
		reward : int
			Reward attained by transition.
		Success : bool
			True if given action was valid action.		

		Parameters
		----------
		actions : int
			Index of action to be performed.
		'''
		raise NotImplementedError()

	@abstractmethod
	def S0(self):
		
		raise NotImplementedError()

	@abstractmethod
	def TICK(self):
		raise NotImplementedError()
		
class Board(MDP):
	def rotatedRight(self, coords):
		result = np.zeros_like(coords)
		result[:, 0], result[:, 1] = -coords[:, 1], coords[:, 0]
		return result
	
	def T(self, action):
		'''
		Make a transition with given action.

		Returns
		----------
		reward : int
			Reward attained by transition.
		Success : bool
			True if given action was valid action.		

		Parameters
		----------
		actions : int
			Index of action to be performed.
		'''
		if action == Action.LEFT:
			if self._tryMove(self.pieces[0], self.piececoords, self.curX-1, self.curY):
				return Reward.MOVEMENT, True
			else:
				return Reward.INVALID_MOVEMENT, False

		elif action == Action.RIGHT:
			if self._tryMove(self.pieces[0], self.piececoords, self.curX+1, self.curY):
				return Reward.MOVEMENT, True
			else:
				return Reward.INVALID_MOVEMENT, False

		elif action == Action.UP:
			if self._tryMove(self.pieces[0], self.rotatedRight(self.piececoords),  self.curX, self.curY):
				return Reward.MOVEMENT, True
			else:
				return Reward.INVALID_MOVEMENT, False
			
		elif action == Action.DOWN:
			return self._oneLineDown()
			
		elif action == Action.SPACE:
			success = True
			while success:
				score, success = self._oneLineDown()
			return score, True

		elif action == Action.STEP:
			return self.TICK()

		else:
			print("Invalid action :",action)

	def TICK(self):
		'''
		Increase time step of MDP.

		Returns
		----------
		reward : int
			Reward attained by transition.
		Success : bool
			True if given action was valid action.		

		Parameters
		----------
		None
		'''
		self.t += 1
		if self.t == self.maxTick:
			return self._gameover(), False
		if self.t % self.TICKS_FOR_LINEDOWN == 0:
			return self.T(Action.DOWN), True
		return Reward.STEP, True

	def S0(self):
		self.table = np.zeros((Board.height, Board.width), dtype='int32')
		#self.table[0,:] = 1
		#self.table[0,0] = 0
		self.curX = 0
		self.curY = 0

		self.t=0		
		self.isOver = False
		
		self.pieces = [0] * 6 # pieces[0] : current, [1] : next, [2] : next2, ...
		self.piececoords = np.zeros((4,2), dtype='int32')
		for i in range(self.N + 1):
			self.nextpiece()

	###########################################################################
	
	def _oneLineDown(self):
		if self._tryMove(self.pieces[0], self.piececoords, self.curX, self.curY-1): #true if downed.
			return Reward.MOVEMENT, True
		else:
			Xs = self.curX + self.piececoords[:, 0]
			Ys = self.curY - self.piececoords[:, 1]
			#print([(x,y) for x,y in zip(Xs,Ys)], self.pieces[0])
			try:
				self.table[Ys,Xs] = self.pieces[0]
			except:
				print(self.table)
				print([(x,y) for x,y in zip(Xs,Ys)], self.pieces[0])
				print(self.piececoords)
				exit()

			#remove full lines
			rowsToRemove = np.where(np.all(self.table, axis=1))[0]
			numsToRemove = len(rowsToRemove)

			if numsToRemove != 0:
				ind = np.delete(np.arange(Board.height), rowsToRemove)
				self.table = np.pad(self.table[ind], ((0,numsToRemove), (0,0)), mode='constant')
				self.interface.line_removed(numsToRemove)
				print("Discovery!")

			gameoverscore = self.nextpiece()
			if gameoverscore:
				return gameoverscore, False
			return numsToRemove, False

	def _canMove(self, shape, coords, newX, newY):
		'''
		Check if piece can be moved to (newX, newY).

		Returns
		----------
		result : bool
			Return true if the movement is possible.
		
		Parameters
		----------
		shape : int
		coords : 2-D numpy array
		newX : int
		newY : int
		'''
		Xs = newX + coords[:, 0]
		Ys = newY - coords[:, 1]
		
		if not (np.all(0<=Xs) and np.all(Xs < Board.width)):
			return False
		if not (np.all(0<=Ys) and np.all(Ys < Board.height)):
			return False
		if np.any(self.table[Ys,Xs] != Tetrominoes.NoShape):
			return False
		return True

	def _tryMove(self, shape, coords, newX, newY):
		'''
		Try to move piece to (newX, newY). If succeed, return True. Otherwise, return False.

		Returns
		----------
		result : bool
			Return true if the movement was successful.
		
		Parameters
		----------
		pieceshape : int
		piececoords : 2-D numpy array
		newX : int
		newY : int
		'''
		if not self._canMove(shape, coords, newX, newY):
			return False
		else:
			self.pieces[0] = shape
			self.piececoords[:] = coords[:]
			self.curX, self.curY = newX, newY
			self.interface.Refresh()
			#print(self.table)
			return True
			
	def nextpiece(self):
		self.pieces = self.pieces[1:]
		newshape = self.interface.newshape() if self.interface else Tetrominoes.NoShape		
		self.pieces.append(newshape)

		# put pieces[0] on stateboard.
		if not self.pieces[0]:
			return 0
		self.piececoords = Tetrominoes.coordsTable[self.pieces[0]].copy()
		self.curX = Board.width // 2 + 1
		self.curY = Board.height - 1 + self.piececoords[:, 1].min()
		

		a = self._tryMove(self.pieces[0], self.piececoords, self.curX, self.curY)
		if not a:
			return self._gameover()
		else:
			return 0

	###########################################################################
	
	width = 10
	height = 22
	def __init__(self, interface):
		self.TICKS_FOR_LINEDOWN=10
		self.N = 5

		self.interface = interface
		self.maxTick = -1
		
	def S(self):
		return (self.table.copy(), list(self.pieces), self.piececoords.copy(), self.curX, self.curY, self.t, self.isOver)

	def setS(self, other):
		state, pieces, piececoords, self.curX, self.curY, self.t, self.isOver = other
		self.table[:] = state[:]
		self.pieces = list(pieces)
		self.piececoords[:] = piececoords[:]

			
	def _gameover(self):
		self.isOver=True
		return Reward.GAMEOVER
