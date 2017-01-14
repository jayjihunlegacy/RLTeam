import datetime
import _pickle
import time
import os
import re

from ..Board import *
from ..Authen import *

class InputType:
	"""
	:class:`InputType` defines possible ways to put input to :class:`BoardInterface`.

	Possible input types are:
	
	1. :data:`MACHINE`
	2. :data:`HUMAN`
	3. :data:`SAVE`
	"""
	MACHINE = 0
	HUMAN = 1
	SAVE = 2
	type2code = {
		'human' : HUMAN,
		'machine' : MACHINE,
		'save' : SAVE}
	tostr = {
		MACHINE : 'Machine',
		HUMAN : 'Human',
		SAVE : 'Save'
		}

class BoardInterface:
	"""
	:class:`BoardInterface` defines basic functionalities as interface. No visual module is implemented.
	To use visual functionalities, use :class:`VisualInterface`.

	:param str input_type: type of input. 'machine', 'human', or 'save'
	:param machine: machine to play when *input_type* is 'machine'
	:type machine: :class:`Tetris.mods.Machines.Machine.Machine`
	"""
	def __init__(self, input_type='machine', machine=None, **kwargs):		
		self.board = Board(self)
		""":class:`Tetris.mods.Board.Board` instance which to interact with."""
		self.machine = machine
		""":class:`Tetris.mods.Machines.Machine` instance which may play Tetris.
		None if no machine is provided."""
		self.isStarted = False
		"""True if started. False if stopped."""
		self.isOver = False
		"""True is over. False otherwise."""
		self.numLinesRemoved = 0
		"""Total number of lines removed since beginning. Starts from 0."""
		self.actionhistory = []
		"""List of actions taken. (tick, action) pairs are stored."""
		self.piecehistory = []
		"""List of spawned pieces. (tick, shape) pairs are stored."""
		self.trajectory = []
		"""List of state. (S,A,R,S') quadruples are stored."""
		self.G = 0
		"""Return so far. Starts from 0."""
		self.settings = dict()
		"""Dictionary of settings include followings:

		1. save_hist
		2. save_traj
		3. collect_traj
		4. collect_hist
		5. verbose_gameover
		6. input_type
		7. filename
		8. name

		#TODO
		explanation about all the settings.
		"""

		#settings
		self.settings['save_hist'] = False
		self.settings['save_traj'] = False
		self.settings['collect_traj'] = False
		self.settings['collect_hist'] = False
		self.settings['verbose_gameover'] = False
		self.settings['input_type'] = InputType.type2code[input_type]
		self.settings['filename'] = 'Nopath'		
		self.settings['name'] = 'Noname'

		for key in kwargs:
			self.settings[key] = kwargs[key]

		self.initialize()
		
	def initialize(self):
		"""
		:meth:`initialize` initializes interface.

		1. clear :attr:`actionhistory`, :attr:`piecehistory`, :attr:`trajectory`.
		2. reset :attr:`numLinesRemoved`, :attr:`G`.
		3. set :attr:`board` to start state(S0) by calling :meth:`Tetris.mods.Board.Board.S0`.
		"""
		self.isStarted = False
		self.isOver = False
		self.numLinesRemoved = 0
		self.actionhistory = []
		self.piecehistory = []
		self.trajectory = []
		self.G = 0
		self.board.S0()
		
	###########################################################################

	def load_savefile(self):
		"""
		:meth:`load_savefile` loads :attr:`actionhistory` and 
		:attr:`piecehistory` from :attr:`settings['filename']`.
		
		This is supposed to be called only when :attr:`settings['input_type']` is :data:`InputType.MACHINE`.
		"""
		env=Env()
		try:
			with open(env.savefolder+self.settings['filename'], 'rb') as f:
				self.actionhistory=_pickle.load(f)
				self.piecehistory=_pickle.load(f)
		except Exception as e:
			print('Failed Loading :',self.settings['filename'])
			print(e)
			exit()

	def save_history(self):
		"""
		:meth:`save_history` saves :attr:`actionhistory` and :attr:`piecehistory` to file.
		
		For example, the file name is Human_150_2016-12-20_13_22-05-123412.sav
		"""
		env = Env()
		now = datetime.datetime.now()
		time_string = str(now).replace(' ','_').replace('.',':').replace(':','-')
		filename = InputType.tostr[self.settings['input_type']] + \
			'_'+str(self.numLinesRemoved)+'_'+time_string+'.sav'
		
		# 4. Save playfile.
		full=env.savefolder+filename
		with open(full, 'wb') as f:
			_pickle.dump(self.actionhistory,f)
			_pickle.dump(self.piecehistory,f)

		print('Saved')
			
	def save_trajectory(self):
		"""
		:meth:`save_trajectory` saves :attr:`trajectory` to file.
		"""
		env = Env()
		folder = env.datafolder
		files = os.listdir(folder)
		filename = 'data.pkl'
		try:
			with open(os.path.join(folder, filename), 'rb') as f:
				d = _pickle.load(f)
		except:
			d = list()

		if len(d) >= 200000:
			files.sort()
			#rename data.pkl to another.
			nums = [int(re.findall(r'\d+', file)[0]) for file in files[1:]]
			nextnum = 1 if len(nums) == 0 else np.max(nums)+1
			newfilename = 'data'+str(nextnum)+'.pkl'
			os.rename(os.path.join(folder, filename), os.path.join(folder, newfilename))

			with open(os.path.join(folder, filename), 'wb') as f:
				_pickle.dump(self.trajectory, f)
		else:
			d = d + self.trajectory
			with open(os.path.join(folder, filename), 'wb') as f:
				_pickle.dump(d, f)

	###########################################################################
	
	def OnTimer(self, event):
		"""
		:meth:`OnTimer` is called by :attr:`timer`, and it ticks the MDP by calling :meth:`tick`.
		"""
		self.tick()
	
	def tick(self):
		"""
		:meth:`tick` increases a timestep of MDP.

		If the game is over, call :meth:`gameover`.

		Otherwise, call :meth:`OnTick`.
		"""
		r, _ = self.board.T(Action.STEP)
		self.G += r

		if self.board.isOver:
			self.gameover()
		else:
			self.OnTick()

	def OnTick(self):
		"""
		:meth:`OnTick` samples an action from :attr:`machine` and perform it by calling :meth:`perform_action`.
		"""
		pi = self.machine.Pi(self.board.S())
		if pi is not None:
			self.perform_action(np.random.choice(5, p=pi))

	def newshape(self):
		"""
		:meth:`newshape` gives new shape.
	   
		If :attr:`settings['collect_hist']` is On, save it to :attr:`piecehistory`.

		:return newshape: integer in [1,7]
		:rtype: int
		"""
		newshape = np.random.randint(1,8)
		if self.settings['collect_hist']:
			self.piecehistory.append((self.board.t, newshape))
		return newshape

	def gameover(self):
		"""
		:meth:`gameover` unsets :attr:`isOver`, and thus stops MDP.

		If :attr:`settings['save_hist']` is On, call :meth:`save_history`.

		If :attr:`settings['save_traj']` is On, call :meth:`save_trajectory`.
		"""
		self.board.pieces[0] = Tetrominoes.NoShape

		self.isStarted=False
		self.isOver = True

		if self.settings['save_hist']:
			self.save_history()

		if self.settings['save_traj']:
			self.save_trajectory()
		
	def perform_action(self, action):
		"""
		:meth:`perform_action` performs *action* by calling :meth:`Tetris.mods.Board.Board.T`.

		Accumulate resulting reward to :attr:`G`.

		If :attr:`settings['collect_hist']` is On, save (tick, action) to :attr:`actionhistory`.

		If :attr:`settings['collect_traj']` is On, save (s, a, r, s') pair to :attr:`trajectory`.

		:param action: action to perform
		:type action: int
		"""
		if self.settings['collect_hist']:
			self.actionhistory.append((self.board.t, action))

		if self.settings['collect_traj']:
			s = self.board.phi()
			a = action
			r, _ = self.board.T(action)
			self.G += r
			sprime = self.board.phi()	
			e = (s,a,r,sprime)
			self.trajectory.append(e)

		else:
			r, _ = self.board.T(action)
			self.G += r
			
	def start(self):
		"""
		#TODO
		"""
		self.initialize()

		self.isStarted=True
		startTime = time.time()
		while True:
			self.OnTimer(None)
			
			if self.isOver:
				break

		endTime=time.time()
		if self.settings['verbose_gameover']:
			print('Game over. Score : %i, Ticks : %i, Time elapsed : %.1fs'%(self.numLinesRemoved, self.board.t, endTime-startTime))
		return self.numLinesRemoved

	###########################################################################

	def line_removed(self, num):
		"""
		:meth:`line_removed` adds :attr:`numLinedRemoved` given *num*.

		:param int num: number of lines newly removed
		"""
		self.numLinesRemoved+=num

	def Refresh(self):
		"""
		:meth:`Refresh` does nothing.
		"""
		pass
		

class DummyInterface(BoardInterface):
	"""
	:class:`DummyInterface` is an :class:`BoardInterface` that has no additional functionalities.

	This class is usually as an interface of dummy board which is used for simulation in MC machines.
	"""
	def __init__(self):
		super().__init__()

	def gameover(self):
		"""
		:meth:`gameover` does nothing.
		"""
		pass
		
	def line_removed(self, num):
		"""
		:meth:`line_removed` does nothing.
		"""
		pass
