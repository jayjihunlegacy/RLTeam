
import wx
import _pickle
import numpy as np
import time

from .BoardInterface import *

from ..Board import *
from ..Authen import Env


class TetrisFrame(wx.Frame):
	def __init__(self):
		wx.Frame.__init__(self, None, title='Tetris', size=(36*Board.width, 30*Board.height + 100))
		self.statusbar = self.CreateStatusBar()
		self.statusbar.SetStatusText('0')

class TetrisPanel(wx.Panel):
	def __init__(self, frame, interface):
		wx.Panel.__init__(self, frame, style=wx.WANTS_CHARS)
		self.Bind(wx.EVT_PAINT, self.OnPaint)
		self.interface= interface

		self.squareWidth, self.squareHeight = None,None
		
	def OnPaint(self, event):
		dc = wx.PaintDC(self)

		board_width = self.interface.board.width
		board_height = self.interface.board.height
		if not self.squareWidth:
			self.squareWidth = self.GetClientSize().GetWidth() / board_width
			self.squareHeight = self.GetClientSize().GetHeight() / board_height

		squareWidth = self.squareWidth
		squareHeight = self.squareHeight

		board = self.interface.board

		#draw background
		dc.SetBrush(wx.Brush('#000000'))
		dc.DrawRectangle(1, 1, board_width * squareWidth - 1, board_height * squareHeight - 1)

		curshape=board.pieces[0]
		piececoords = board.piececoords

		#draw dropped pieces
		Ys, Xs = np.nonzero(board.table)
		for x, y in zip(Xs, Ys):
			shape = board.table[y,x]
			self.drawSquare(dc, x * squareWidth, (board_height - y - 1) * squareHeight, shape)

		#draw current piece
		Xs = board.curX + piececoords[:, 0]
		Ys = board.curY - piececoords[:, 1]
		for x, y in zip(Xs, Ys):
			self.drawSquare(dc, x * squareWidth, (board_height - y - 1) * squareHeight, curshape)

		#draw aim
		'''
		if curPiece.shape() != Tetrominoes.NoShape:
			AimY = self.interface.board.curY
			while AimY > 0:
				if not self.interface.board.canMove(curPiece, self.interface.board.curX, AimY - 1):
					break
				AimY-=1

			Xs = self.interface.board.curX + curPiece.Xs()
			Ys = AimY - curPiece.Ys()

			for x, y in zip(Xs, Ys):
				self.drawSquare(dc, x * squareWidth, (board_height - y - 1) * squareHeight, Tetrominoes.Aim)
		'''
		self.interface.set_statusbar('%.3f'%(self.interface.G))

	def drawSquare(self,dc,x,y,shape):
		colors = ['#000000', '#CC6666', '#66CC66', '#6666CC',
                  '#CCCC66', '#CC66CC', '#66CCCC', '#DAAA00',
				  '#646464']

		light = ['#000000', '#F89FAB', '#79FC79', '#7979FC', 
                 '#FCFC79', '#FC79FC', '#79FCFC', '#FCC600',
				 '#969696']

		dark = ['#000000', '#803C3B', '#3B803B', '#3B3B80', 
                '#80803B', '#803B80', '#3B8080', '#806200',
				'#323232']
				
		board_width = self.interface.board.width
		board_height = self.interface.board.height

		if not self.squareWidth:
			self.squareWidth = self.GetClientSize().GetWidth() / board_width
			self.squareHeight = self.GetClientSize().GetHeight() / board_height

		squareWidth = self.squareWidth
		squareHeight = self.squareHeight

		pen = wx.Pen(light[shape])
		pen.SetCap(wx.CAP_PROJECTING)
		dc.SetPen(pen)

		dc.DrawLine(x, y + squareHeight - 1, x, y)
		dc.DrawLine(x, y, x + squareWidth - 1, y)

		darkpen = wx.Pen(dark[shape])
		darkpen.SetCap(wx.CAP_PROJECTING)
		dc.SetPen(darkpen)

		dc.DrawLine(x + 1, y + squareHeight - 1, x + squareWidth - 1, y + squareHeight - 1)
		dc.DrawLine(x + squareWidth - 1, y + squareHeight - 1, x + squareWidth - 1, y + 1)

		dc.SetPen(wx.TRANSPARENT_PEN)
		dc.SetBrush(wx.Brush(colors[shape]))
		dc.DrawRectangle(x + 1, y + 1, squareWidth - 2, squareHeight - 2)

class Visual:
	def __init__(self):
		self.frame = TetrisFrame()
		self.panel = TetrisPanel(self.frame, self)
		
		self.limit_HZ = True
		if self.limit_HZ:
			self.ID_PAINTER = 2
			self.paint_timer = wx.Timer(self.panel, self.ID_PAINTER)
			self.panel.Bind(wx.EVT_TIMER, self.OnHZ, id=self.ID_PAINTER)
			self.refresh = True
			self.paint_timer.Start(60)		

		self.ID_TIMER = 1
		
		self.timer = wx.Timer(self.panel, self.ID_TIMER)		
		
		def prepare():
			self.panel.SetFocus()
			self.frame.Center()
			self.frame.Show(True)

		prepare()

	def OnHZ(self, event):
		if self.refresh:
			self.panel.Refresh()
			self.refresh = False

	def Refresh(self):
		if self.limit_HZ:
			self.refresh = True
		else:
			self.panel.Refresh()
			
	def set_statusbar(self, string):
		self.frame.statusbar.SetStatusText(string)

	def start(self):
		print('Start!')
		self.initialize()

		if self.settings['input_type'] == InputType.SAVE:
			self.load_savefile()

		self.isStarted=True
		self.timer.Start(10)

class VisualInterface(Visual, BoardInterface):
	keycode2action={
		wx.WXK_LEFT : Action.LEFT, 
		wx.WXK_RIGHT: Action.RIGHT, 
		wx.WXK_UP	: Action.UP,
	    wx.WXK_DOWN : Action.DOWN, 
		wx.WXK_SPACE: Action.SPACE}
	
	def __init__(self, **kwargs):
		Visual.__init__(self)
		BoardInterface.__init__(self, **kwargs)

		self.panel.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
		self.panel.Bind(wx.EVT_TIMER, self.tick, id=self.ID_TIMER)
			
	def OnKeyDown(self, event):
		if not self.isStarted and not self.isOver:
			event.Skip()
			return

		keycode = event.GetKeyCode()
		#1. restart
		if keycode in [ord('R'), ord('r')]:
			self.start()
			return
		
		#3-1. when over, ignore movement key.
		if self.isOver:
			return

		#4. when valid key pressed, perform it.
		try:
			action = self.keycode2action[keycode]
			self.perform_action(action)
		except KeyError:
			pass
					
	def gameover(self):
		self.timer.Stop()
		self.set_statusbar('Game Over! ' + '%.3f'%(self.G))
		BoardInterface.gameover(self)

	def perform_action(self, action):
		if action != Action.STEP:
			self.actionhistory.append((self.board.t, action))
		r, _ = self.board.T(action)
		self.G += r
		if action != Action.STEP:
			pass
			#print('Pressed :', Action.tostr[action], r)
		
	def OnTick(self):
		if self.settings['input_type'] == InputType.MACHINE:
			pi = self.machine.Pi(self.board.S())
			if pi is not None:
				try:
					self.perform_action(np.random.choice(5, p=pi))
				except ValueError:
					print('Probabilities do not sum to 1. P =',str(pi))
					raise Exception
			else:
				self.perform_action(Action.STEP)

		elif self.settings['input_type'] == InputType.SAVE:
			while len(self.actionhistory) > 0:
				data = self.actionhistory[0]
				tick, action = data
				if self.board.t == tick:
					self.perform_action(action)
					self.actionhistory = self.actionhistory[1:]

				elif self.board.t > tick:
					print('Problem')
				else:
					self.perform_action(Action.STEP)

		elif self.settings['input_type'] == InputType.HUMAN:
			self.perform_action(Action.STEP)

	def newshape(self):
		if self.settings['input_type'] == InputType.SAVE:
			data = self.piecehistory[0]
			tick, piece = data
			if self.board.t == tick:
				self.piecehistory = self.piecehistory[1:]
				return piece
			elif self.board.t > tick:
				print('Skipped shape.')
				print('Current tick : %i, Data : (%i,%i)'%(self.board.t, tick, piece))
				self.piecehistory = self.piecehistory[1:]
				return piece	
		else:
			return BoardInterface.newshape(self)
