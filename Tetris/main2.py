from Machines.StochasticMachine import StochasticMachine
from Interfaces.BoardInterface import *
from Authen import *
import os
import numpy as np
import _pickle
import time
from Machines.NeuralMachine import CNNMachine
from Interfaces.RLInterface import PGInterface
from functools import reduce
from theano import shared, function, scan
import theano.tensor as T
###############################################################################
#1. build machine
machine = CNNMachine()
machine.compile_model()

print('#1. machine built.')

###############################################################################
#2. ready interface
interface = PGInterface('PGBoard', machine)
params = interface.machine.params

# tensor variables
X = interface.machine.s # (H, 2, 22, 10)
As = T.ivector('As')
pi = interface.machine.pi # pi(X ; params) Shape (1,5)
				
log_pi, _ = scan(lambda p, a:T.log(p[a]), sequences=[pi, As])
g = T.grad(log_pi.sum(), param)
f = function([X, As], g)

# shared variables
interface.Xs_shared = shared(np.zeros((0,0,0,0), dtype='float32'))
interface.As_shared = shared(np.zeros((0,), dtype='int32'))
interface.Rs_shared = shared(np.zeros((0,), dtype='float32'))
interface.lr_shared = shared(np.array(interface.lr , dtype='float32'))
interface.b_shared = shared(np.array(0., dtype='float32'))
interface.functions = []
one_grad = T.grad(T.log(pi[0][0]), params[0])
f2 = function([X], one_grad)
print(np.array(f2(x)[0]))
print(np.array(f2(x[:2])[0]))
print(np.array(f2(x[0].reshape((1,2,22,10)))[0]))
#sum of gradients == gradient of sum!!
#lefts, _ = scan(lambda x:x.reshape((1,2,22,10)), X)
f = function([X], lefts)

print(f(x).shape)
exit(0)
###			
interface.functions = []
for a in range(5):
	one_grad = []
	for param in params:
		one_grad.append(T.grad(T.log(pi[0][a]), param))
	interface.functions.append(function([X], one_grad))

print('#2. interface ready.')

###############################################################################
#3. one roll-out
interface.initialize()
interface.isStarted=True
interface.board.S0()
interface.board.nextpiece()
while True:
	interface.OnTimer(None)
	if interface.isOver:
		break
		
print('#3. roll-out made.')
###############################################################################
#4. learn
Xs = np.array([e[0] for e in interface.experiences], dtype='float32')
As = np.array([e[1] for e in interface.experiences], dtype='int32')
Rs = np.array([e[2] for e in interface.experiences], dtype='float32')
b = 0
		
interface.Xs_shared.set_value(Xs)
interface.As_shared.set_value(As)
interface.Rs_shared.set_value(Rs)
interface.b_shared.set_value(b)

print('#4-1. shared variables set.')
print('Xs shape :',Xs.shape)
				
del_params = []
for i, a in enumerate(As):
	f = interface.functions[a]
	per_params = f(Xs[i].reshape((1,2,22,10)))
	del_params.append(per_params)
	
print('#4-2. del_params calculated.')
			
del_params = reduce(lambda X,Y:[x+y for x,y in zip(X,Y)], del_params)
rightsum = np.sum(Rs) - b
del_params = [T.mul(g, rightsum, interface.lr) for g in del_params]
###############################################################################
#update the parameter values!
params = interface.machine.params
for param, del_param in zip(params, del_params):
	param += del_param
	
print('#4-3. everything cleared.')

