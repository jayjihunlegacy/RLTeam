import numpy as np
class StateActionNode:#(s,a) pair..
	def __init__(self, N, parent=None):
		self.N = N
		self.visits = 0
		self.reward = 0.
		self.parent=parent

		self.isroot=False
		self.childs = []
		for i in range(self.N):
			self.childs.append(None)
			
	def get_child(self, action):
		if self.childs[action] is None:
			self.childs[action] = StateActionNode(self.N, self)
		return self.childs[action]
		
	def update(self, reward):
		self.reward += reward
		self.visits +=1

	def get_Qs(self):
		Qs = [child.reward/child.visits if child else -float('inf') for child in self.childs]
		return Qs

	def backprop(self, reward):
		self.update(reward)
		if self.parent:
			self.parent.backprop(reward)

class Node:
	def __init__(self, N, rot=0, parent=None):
		self.rot = rot
		self.N = N
		self.childs = [None] * self.N
		
	def get_child(self, action, rotinc=0):
		if self.childs[action] is None:
			self.childs[action] = Node(self.N, self.rot+rotinc, self)
		return self.mov_child(action)

	def mov_child(self, action):		
		return self.childs[action]

	def avail_actions(self, maxrot):
		actions = np.where([True if child and child.rot <= maxrot else False \
			for child in self.childs])[0]
		return actions