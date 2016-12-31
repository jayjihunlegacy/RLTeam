from Board import *
from Authen import *
import datetime
import _pickle
import time
import os
import re

class InputType:
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
	def __init__(self, input_type='machine', machine=None, **kwargs):
		self.board = Board(self)
		self.machine = machine
		self.settings = dict()

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
		self.isStarted = False
		self.isOver = False
		self.numLinesRemoved = 0

		self.actionhistory = []
		self.piecehistory = []
		self.trajectory = []
		self.R = 0
		self.board.S0()

	def load_savefile(self):
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

	def OnTimer(self, event):
		self.tick()
	
	def tick(self):
		r, _ = self.board.T(Action.STEP)
		self.OnStepreward(r)

		if self.board.isOver:
			self.gameover()
		else:
			self.OnTick()

	def OnTick(self):
		pi = self.machine.Pi(self.board.S())
		if pi is not None:
			self.perform_action(np.random.choice(5, p=pi))

	def OnStepreward(self, reward):
		pass

	def newshape(self):
		newshape = np.random.randint(1,8)
		if self.settings['collect_hist']:
			self.piecehistory.append((self.board.t, newshape))
		return newshape

	def gameover(self):
		self.board.pieces[0] = Tetrominoes.NoShape

		self.isStarted=False
		self.isOver = True

		if self.settings['save_hist']:
			self.save_history()

		if self.settings['save_traj']:
			self.save_trajectory()
		
	def line_removed(self, num):
		self.numLinesRemoved+=num

	def perform_action(self, action):
		if self.settings['collect_hist']:
			self.actionhistory.append((self.board.t, action))

		if self.settings['collect_traj']:
			s = self.phi(self.board.S())
			a = action
			r, _ = self.board.T(action)
			self.R += r
			sprime = self.phi(self.board.S())		
			e = (s,a,r,sprime)
			self.trajectory.append(e)

		else:
			r, _ = self.board.T(action)
			self.R += r

	def phi(self, S):
		board, pieces, piececoords, curX, curY, ticks, isOver = S
		
		x_piece = np.zeros((Board.height, Board.width), dtype='int32')
		Xs = curX + piececoords[:, 0]
		Ys = curY - piececoords[:, 1]
		x_piece[Ys,Xs]=1		
		X = np.stack([(board>0).astype('int32'), x_piece])		
		return X

	def Refresh(self):
		pass
		
	def start(self):
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

	def save_trajectory(self):
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
	
class DummyInterface(BoardInterface):
	def __init__(self):
		super().__init__()

	def gameover(self):
		pass
		
	def line_removed(self, num):
		pass
