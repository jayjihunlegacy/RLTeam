import _pickle
import numpy as np
import time
from abc import ABCMeta, abstractmethod

class Action:
	"""
	:class:`Action` defines a set of actions that can be performed.

	Actions involves :

	+--------------+-----------------------------+
	|Action        |Effect                       |
	+==============+=============================+
	|:data:`LEFT`  |Move a piece to left         |
	+--------------+-----------------------------+
	|:data:`RIGHT` |Move a piece to right        |
	+--------------+-----------------------------+
	|:data:`UP`    |Rotate a piece clockwise     |
	+--------------+-----------------------------+
	|:data:`DOWN`  |Move a piece down            |
	+--------------+-----------------------------+
	|:data:`SPACE` |Drop a piece                 |
	+--------------+-----------------------------+
	|:data:`STEP`  |**NOT EXECUTABLE**           |
	+--------------+-----------------------------+
	
	:data:`STEP` action is not executable by agent, but automatically called by :class:`BoardInterface`.
	"""

	LEFT = 0
	"""
	:data:`LEFT` moves a piece to left.
	"""
	RIGHT = 1
	"""
	:data:`RIGHT` moves a piece to right.
	"""
	UP = 2
	"""
	:data:`UP` rotates a piece clockwise.
	"""
	DOWN = 3
	"""
	:data:`DOWN` moves a piece down.
	"""
	SPACE = 4
	"""
	:data:`SPACE` drops a piece.
	"""
	STEP = 5
	"""
	:data:`STEP` increases timestep. It is not executable.
	"""
	tostr={
		LEFT :	'LEFT',
		RIGHT :	'RIGHT',
		UP :	'UP',
		DOWN :	'DOWN',
		SPACE :	'SPACE'
	}
	"""
	:data:`tostr` converts action index to its action name.
	"""

class Reward:
	"""
	:class:`Reward` defines a set of reward that is given to agent when action is performed at each state.

	+------------------------+------+
	|Transition              |Reward|
	+========================+======+
	|:data:`MOVEMENT`        |0     |
	+------------------------+------+
	|:data:`INVALID_MOVEMENT`|0     |
	+------------------------+------+
	|:data:`LINECLEAR`       |1     |
	+------------------------+------+
	|:data:`STEP`            |0     |
	+------------------------+------+
	|:data:`GAMEOVER`        |0     |
	+------------------------+------+
	"""

	MOVEMENT = 0
	"""
	:data:`MOVEMENT` is a reward given whenever any action is taken.
	If a reward for more specific description exisits, it overrides :data:`MOVEMENT`.
	"""
	INVALID_MOVEMENT = 0
	"""
	:data:`INVALID_MOVEMENT` is a reward given whenever any invalid action is taken.
	"""
	LINECLEAR = 1
	"""
	:data:`LINECLEAR` is a reward given each time a line is cleared.
	"""
	STEP = 0
	"""
	:data:`STEP` is a reward given each time a timestep advances.
	It is naturally set to zero.
	"""
	GAMEOVER = 0
	"""
	:data:`GAMEOVER` is a reward given when game is over.
	"""
	
class Tetrominoes:
	"""
	:class:`Tetrominoes` defines possible shapes of a piece and its coordinates.

	Possible shapes are:

	1. :data:`ZShape`
	2. :data:`SShape`
	3. :data:`LineShape`
	4. :data:`TShape`
	5. :data:`SquareShape`
	6. :data:`LShape`
	7. :data:`MirroredLShape`
	"""
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
	"""
	:class:`MDP` is an abstract base class of MDP.

	:meth:`T`, :meth:`S0`, :meth:`TICK` should be implemented.
	"""
	__metaclass__ = ABCMeta
	@abstractmethod
	def T(self, action):
		'''
		:meth:`T` makes a transition with given action.

		:param int action: Index of action to be performed.

		:return (reward, success): *reward* is a reward attained by transition. *success* is true if given action was valid action.	
		:rtype: (int, bool)
		'''
		raise NotImplementedError()

	@abstractmethod
	def S0(self):
		'''
		:meth:`S0` sets an MDP to a start state.
		'''		
		raise NotImplementedError()

	@abstractmethod
	def TICK(self):
		"""
		:meth:`TICK` increases timestep by 1.
		"""
		raise NotImplementedError()
		
class Board(MDP):
	"""
	:class:`Board` is a class representing Tetris in MDP form.
	"""

	width = 10
	"""
	Width of Tetris board
	"""

	height = 22
	"""
	Height of Tetris board
	"""
	def __init__(self, interface):
		self.TICKS_FOR_LINEDOWN=10
		self.N = 5

		self.interface = interface
		self.maxTick = -1
		
	def T(self, action):
		'''
		:meth:`T` makes a transition with given action.

		:param int action: Index of action to be performed.

		:return (reward, success): *reward* is a reward attained by transition. *success* is true if given action was valid action.	
		:rtype: (int, bool)
		'''
		if action == Action.LEFT:
			if self.tryMove(self.pieces[0], self.piececoords, self.curX-1, self.curY):
				return Reward.MOVEMENT, True
			else:
				return Reward.INVALID_MOVEMENT, False

		elif action == Action.RIGHT:
			if self.tryMove(self.pieces[0], self.piececoords, self.curX+1, self.curY):
				return Reward.MOVEMENT, True
			else:
				return Reward.INVALID_MOVEMENT, False

		elif action == Action.UP:
			if self.tryMove(self.pieces[0], self._rotatedRight(self.piececoords),  self.curX, self.curY):
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
		"""
		:meth:`TICK` increases timestep by 1.

		:return (reward, success): *reward* is a reward attained by transition. *success* is true if given action was valid action.	
		:rtype: (int, bool)
		"""
		self.t += 1
		if self.t == self.maxTick:
			return self._gameover(), False
		if self.t % self.TICKS_FOR_LINEDOWN == 0:
			return self.T(Action.DOWN)
		return Reward.STEP, True

	def S0(self):
		'''
		:meth:`S0` sets an MDP to a start state.

		#TODO explain more in detail.
		'''		
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

	def _rotatedRight(self, coords):
		result = np.zeros_like(coords)
		result[:, 0], result[:, 1] = -coords[:, 1], coords[:, 0]
		return result
	
	def _oneLineDown(self):
		if self.tryMove(self.pieces[0], self.piececoords, self.curX, self.curY-1): #true if downed.
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

			gameoverscore = self.nextpiece()
			if gameoverscore:
				return gameoverscore, False
			return numsToRemove, False

	def canMove(self, shape, coords, newX, newY):
		'''
		:meth:`canMove` checks if piece can be moved to (*newX*, *newY*).

		:param int shape: shape of a piece to be placed
		:param np.ndarray coords: (4,2) int32 numpy array. it includes coordinates of \
		four blocks in a piece to be placed
		:param int newX: New X value to be placed
		:param int newY: New Y value to be placed

		:return result: True if movement is possible
		:rtype: bool
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

	def tryMove(self, shape, coords, newX, newY):
		"""
		:meth:`tryMove` moves piece if possible. If impossible, do nothing.

		:param int shape: shape of a piece to be placed
		:param np.ndarray coords: (4,2) int32 numpy array. it includes coordinates of \
		four blocks in a piece to be placed
		:param int newX: New X value to be placed
		:param int newY: New Y value to be placed

		:return result: True if movement is possible
		:rtype: bool
		"""
		if not self.canMove(shape, coords, newX, newY):
			return False
		else:
			self.pieces[0] = shape
			self.piececoords[:] = coords[:]
			self.curX, self.curY = newX, newY
			self.interface.Refresh()
			#print(self.table)
			return True
			
	def nextpiece(self):
		"""
		:meth:`nextpiece` spawns new piece and place next piece on the board.

		New piece will be given by :attr:`self.interface` and be appended at the end of :attr:`self.pieces`.
		If next piece cannot be placed to proper position, the game is over.

		:returns reward: a reward given by transition
		:rtype: int or float
		"""
		self.pieces = self.pieces[1:]
		newshape = self.interface.newshape() if self.interface else Tetrominoes.NoShape		
		self.pieces.append(newshape)

		# put pieces[0] on stateboard.
		if not self.pieces[0]:
			return 0
		self.piececoords = Tetrominoes.coordsTable[self.pieces[0]].copy()
		self.curX = Board.width // 2 + 1
		self.curY = Board.height - 1 + self.piececoords[:, 1].min()
		

		a = self.tryMove(self.pieces[0], self.piececoords, self.curX, self.curY)
		if not a:
			return self._gameover()
		else:
			return 0

	###########################################################################
			
	def phi(self):
		"""
		:meth:`phi` encodes state tuple into (2, 22, 10) tensor.
		
		:return phi: binary (2, 22, 10) state tensor
		:rtype: numpy.ndarray
		"""
		x_piece = np.zeros((Board.height, Board.width), dtype='int32')
		Xs = self.curX + self.piececoords[:, 0]
		Ys = self.curY - self.piececoords[:, 1]
		x_piece[Ys,Xs]=1		
		X = np.stack([(self.table>0).astype('int32'), x_piece])

		return X

	def S(self):
		"""
		:meth:`S` returns a current state of MDP in a tuple.

		Tuple contains:

		1. Copied matrix of :attr:`table`
		2. Copied list of :attr:`pieces`
		3. Copied matrix of :attr:`piececoords`
		4. :attr:`curX`
		5. :attr:`curY`
		6. :attr:`t`
		7. :attr:`isOver`

		:returns state: current state in a tuple form
		:rtype: tuple
		"""
		return (self.table.copy(), list(self.pieces), self.piececoords.copy(), self.curX, self.curY, self.t, self.isOver)

	def setS(self, other):
		"""
		:meth:`setS` sets MDP's state to given *other*.

		The format of *other* should be the same as that returned by :meth:`S`.

		:param tuple other: state to be set		
		"""
		state, pieces, piececoords, self.curX, self.curY, self.t, self.isOver = other
		self.table[:] = state[:]
		self.pieces = list(pieces)
		self.piececoords[:] = piececoords[:]

			
	def _gameover(self):
		self.isOver=True
		return Reward.GAMEOVER
