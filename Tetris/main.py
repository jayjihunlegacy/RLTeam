import os
import numpy as np
import _pickle
import time
import traceback

from mods.Machines.Machine import *
from mods.Machines.MCMachine import *
from mods.Interfaces.BoardInterface import *
from mods.Authen import *

class Officer:
	def main_loop(self):
		while True:
			print('''Choose option.
1. Human Play.
 11. Saved Play.
 12. CNNPolicyMachine Play.
 13. CNNValueMachine Play.
 14. StochasticMachine2
 15. MCTC Play.
 16. micro MC Play.
 17. macro MC Play. 

2. CNNValueMachine infinite Value training

4. Generate Demos.

5. any...

6. Train CNNPolicyMachine.
 61. Train CNNValueMachine.

7. Q-learning RL
 71. DQN RL
 72. Double-DQN RL
 73. SARSA RL
 74. Expected Sarsa RL
 
8. Vanilla Policy Gradient
 81. PGT Policy Gradient
 82. Actor-Critic Policy Gradient


9. evaluate
 91. evaluate MCMachine
 92. evaluate MCMachine with maxtime .1s, .2s, .5s
 93. evaluate MCMachine with maxstep 10, 20, 30, 40, 
 94. evaluate CNNValueMachine
 95. evaluate CNNPolicyMachine

0. Quit''')
			option = int(input('>>'))
			if option==1:
				self.visual_play('human')
				
			elif option==11:
				filename = 'Human_27_2016-12-10_19-33-10-090293.sav'
				self.visual_play('save',filename=filename)
				
			elif option==12:
				from mods.Machines.NeuralMachine import CNNPolicyMachine
				machine = CNNPolicyMachine()
				machine.compile_model()
				self.visual_play('machine', machine=machine)

			elif option==13:
				from mods.Machines.NeuralMachine import CNNValueMachine
				machine = CNNValueMachine()
				self.visual_play('machine', machine=machine)
				
			elif option==14:
				machine = StochasticMachine2(0)
				self.visual_play('machine', machine=machine)

			elif option==15:
				machine = MCTSMachine()
				self.visual_play('machine', machine=machine)

			elif option==16:
				machine = MicroMCMachine()
				self.visual_play('machine', machine=machine)

			elif option==17:
				machine = MacroMCMachine()
				self.visual_play('machine', machine=machine)
			
			elif option==2:
				machine = CNNValueMachine()

			elif option==4:
				self.generate_demos()

			elif option==5:
				pass

			elif option==6:
				self.train_cnnpolicy()
			elif option==61:
				self.train_cnnvalue()

			elif option==7:
				from mods.Machines.NeuralMachine import CNNValueMachine
				from mods.Interfaces.RLInterface import ValueRLInterface
				machine = CNNValueMachine(name='CNNValueQ-LearningMachine',
							  batch_size=8192,
							  v_to_pi='e-greedy',
							  epsilon=0.2)
				interface = ValueRLInterface(machine,
								 rlmode='q-learning', 
								 lr=0.01,
								 maxtick=10000)
				interface.train()

			elif option==71:
				from mods.Machines.NeuralMachine import CNNValueMachine
				from mods.Interfaces.RLInterface import ValueRLInterface
				machine = CNNValueMachine(name='CNNValueDQNMachine',
							  batch_size=8192,
							  v_to_pi='e-greedy',
							  epsilon=0.2)
				interface = ValueRLInterface(machine,
								 rlmode='dqn', 
								 lr=0.01,
								 maxtick=10000)
				interface.train()

			elif option==73:
				from mods.Machines.NeuralMachine import CNNValueMachine
				from mods.Interfaces.RLInterface import ValueRLInterface
				machine = CNNValueMachine(name='CNNValueSARSAMachine',
							  batch_size=8192,
							  v_to_pi='e-greedy',
							  epsilon=0.2)
				interface = ValueRLInterface(machine,
								 rlmode='sarsa', 
								 lr=0.01,
								 maxtick=10000)
				interface.train()

			elif option==74:
				from mods.Machines.NeuralMachine import CNNValueMachine
				from mods.Interfaces.RLInterface import ValueRLInterface
				machine = CNNValueMachine(name='CNNValueESARSAMachine',
							  batch_size=8192,
							  v_to_pi='e-greedy',
							  epsilon=0.2)
				interface = ValueRLInterface(machine,
								 rlmode='expected-sarsa', 
								 lr=0.01,
								 maxtick=10000)
				interface.train()
				
			elif option==8:
				from mods.Machines.NeuralMachine import CNNPolicyMachine
				from mods.Interfaces.RLInterface import PGInterface
				machine = CNNPolicyMachine(name='CNNPolicyVanillaPGMachine')
				interface = PGInterface(machine,'vanilla',maxtick=10000)		
				interface.train()

			elif option==81:
				from mods.Machines.NeuralMachine import CNNPolicyMachine
				from mods.Interfaces.RLInterface import PGInterface
				machine = CNNPolicyMachine(name='CNNPolicyPGTPGMachine')
				interface = PGInterface(machine,'pgt',maxtick=10000)		
				interface.train()

			elif option==82:
				from mods.Machines.NeuralMachine import CNNPolicyMachine
				from mods.Interfaces.RLInterface import PGInterface
				#machine = CNNPolicyMachine()
				interface = PGInterface(machine,'actorcritic',maxtick=10000)		
				interface.train()

			elif option==9:
				self.evaluate()

			elif option==91:
				machine = MicroMCMachine()
				score = machine.evaluate()
				print('Score :',score)
				
			elif option==92:
				machine = MicroMCMachine(maxtime=.1)
				score = machine.evaluate(200)
				print('Score of .1s :',score)#result : (32.78, 123,15)
				
				machine = MicroMCMachine(maxtime=.2)
				score = machine.evaluate(200)
				print('Score of .2s :',score)#result : (33.55, 135.69)
				
				machine = MicroMCMachine(maxtime=.5)
				score = machine.evaluate(200)
				print('Score of .5s :',score)#result : (34.08, 145.45)
				
			elif option==93:
				machine = MicroMCMachine(maxstep=10, maxtime=.1)
				score = machine.evaluate(200)
				print('Score of 10 :',score)#result : (26.48, 104.4)
				
				machine = MicroMCMachine(maxstep=20, maxtime=.1)
				score = machine.evaluate(200)
				print('Score of 20 :',score)#result : (30.52, 135.54)
				
				machine = MicroMCMachine(maxstep=30, maxtime=.1)
				score = machine.evaluate(200)
				print('Score of 30 :',score)#result : (33.38, 145.38)
				
				machine = MicroMCMachine(maxstep=40, maxtime=.1)
				score = machine.evaluate(200)
				print('Score of 40 :',score)#result : (32.80, 115.09)
				
				machine = MicroMCMachine(maxstep=50, maxtime=.1)
				score = machine.evaluate(200)
				print('Score of 50 :',score)#result : (32.37, 140.87)

			elif option==94:
				from mods.Machines.NeuralMachine import CNNValueMachine
				machine = CNNValueMachine()
				score = machine.evaluate(10000)
				print('Score :',score)
				
			elif option==0:
				break
			
	def visual_play(self, input_type, machine=None, filename=None):
		env = Env()
		if env.use_wx:
			from mods.Interfaces.VisualInterface import VisualInterface
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
		score = machine.evaluate()
		print('Score :',score)
		
	def generate_demos(self,num_of_demo=1000):
		np.random.seed(int(time.time()))
		try:	
			teacher = StochasticMachine(0.01)
			interface = BoardInterface(machine=teacher, collect_traj=True, save_traj=True)
			for i in range(num_of_demo):
				interface.start()
		except:
			print(traceback.format_exc())
			
	def what(self):
		from mods.Machines.NeuralMachine import CNNPolicyMachine
		machine = CNNPolicyMachine()
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
			

	def train_cnnpolicy(self):
		from mods.Machines.NeuralMachine import CNNPolicyMachine
		machine = CNNPolicyMachine()
		machine.train()
		
	def train_cnnvalue(self):
		from mods.Machines.NeuralMachine import CNNValueMachine
		machine = CNNValueMachine()
		machine.train()
			
def main():
	officer = Officer()
	officer.main_loop()
	

if __name__=='__main__':
	main()
