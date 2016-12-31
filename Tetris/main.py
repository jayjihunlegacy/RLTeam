from Machines.Machine import *
from Machines.MCMachine import *
from Interfaces.BoardInterface import *
from Authen import *
import os
import numpy as np
import _pickle
import time
import traceback

class Officer:
	def main_loop(self):
		while True:
			print('''Choose option.
1. Human Play.
2. Saved Play.
3. Neural Play.
4. Heuristic Play.
 41. StochasticMachine2

 42. MCTC Play.
 43. micro MC Play.
 44. macro MC Play.
 

5. Generate Demos.
6. any...
7. Train Neural Agent.
8. PG RL
9. evaluate
0. Quit''')
			option = int(input('>>'))
			if option==1:
				self.visual_play('human')
				
			elif option==2:
				filename = 'Human_27_2016-12-10_19-33-10-090293.sav'
				self.visual_play('save',filename=filename)
				
			elif option==3:
				from Machines.NeuralMachine import CNNMachine
				machine = CNNMachine()
				machine.compile_model()
				self.visual_play('machine', machine=machine)
				
			elif option==4:
				machine = StochasticMachine(0)
				self.visual_play('machine', machine=machine)

			elif option==41:
				machine = StochasticMachine2(0)
				self.visual_play('machine', machine=machine)

			elif option==42:
				machine = MCTSMachine()
				self.visual_play('machine', machine=machine)

			elif option==43:
				machine = MicroMCMachine()
				self.visual_play('machine', machine=machine)

			elif option==44:
				machine = MacroMCMachine()
				self.visual_play('machine', machine=machine)
			

			elif option==5:
				self.generate_demos()
			elif option==6:
				self.what()
			elif option==7:
				self.train()
			elif option==8:
				self.PG()

			elif option==9:
				self.evaluate()


			elif option==0:
				break
			
	def visual_play(self, input_type, machine=None, filename=None):
		env = Env()

		if env.use_wx:
			from Interfaces.VisualInterface import VisualInterface
			import wx
			app = wx.App()

			interface = VisualInterface(input_type=input_type, machine=machine, filename=filename)
		else:
			interface = BoardInterface(input_type=input_type, machine=machine, filename=filename)

		interface.start()

		if env.use_wx:
			app.MainLoop()

###############################################################################

	def evaluate(self):
		machine = StochasticMachine(0)
		score = machine.evaluate(10000)
		print('Score :',score)
		
	def generate_demos(self,num_of_demo=10):
		np.random.seed(int(time.time()))			
		try:	
			teacher = StochasticMachine(0.01)
			interface = BoardInterface(machine=teacher, collect_traj=True, save_traj=True)
			for i in range(num_of_demo):
				interface.start()
		except:
			print(traceback.format_exc())
			
	def what(self):
		from Machines.NeuralMachine import CNNMachine
		machine = CNNMachine()
		machine.load_dataset()
		machine.compile_model()
		N = len(machine.X)		
		idx = np.random.randint(N,size=(100,))
		Xs = machine.X[idx]
		Zs = machine.Y[idx]
		Ys = machine.model.predict(Xs)
		for X,Z,Y in zip(Xs,Zs,Ys):
			print(X)
			print(Z)
			print(Y)
			

	def train(self):
		from Machines.NeuralMachine import CNNMachine
		machine = CNNMachine()
		machine.train()

	def PG(self):
		from Machines.NeuralMachine import CNNMachine
		from Interfaces.RLInterface import PGInterface
		machine = CNNMachine()
		machine.compile_model()
		interface = PGInterface('PGBoard', machine)
		
		episode_num=0
		history = {'r':[], 'ticks':[]}
		print('Start Policy Gradient')
		try:
			while True:
				r, ticks = interface.start()	
				history['r'].append(r)
				history['ticks'].append(ticks)		
				episode_num+=1
				if episode_num%100 == 0:
					print('Episode #%i.'%(episode_num,))
					machine.save_weights()
					with open('PG_History.pkl', 'wb') as f:
						_pickle.dump(history, f)
		except:		
			with open('PG_History.pkl', 'wb') as f:
				_pickle.dump(history, f)
			
	def DQN(self):
		from Machines.NeuralMachine import CNNMachine
		from Interfaces.RLInterface import DQNRLInterface
		machine = CNNMachine()
		machine.compile_model()
		interface = DQNRLInterface('DQN', machine)
		
		episode_num=0
		history = {'score':[], 'ticks':[]}
		print("Start DQN!")
		try:
			while True:
				score, ticks = interface.start()			
				history['score'].append(score)
				history['ticks'].append(ticks)
				episode_num+=1
			
				if episode_num%10 == 0:
					print('Episode #%i.'%(episode_num,))
					
		except:
			with open('DQN_History.pkl','wb') as f:
				_pickle.dump(history, f)
def main():
	officer = Officer()
	officer.main_loop()
	

if __name__=='__main__':
	main()
