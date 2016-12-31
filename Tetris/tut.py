import numpy
import theano
import theano.tensor as T
import time
rng = numpy.random

N = 400                                   # training sample size
feats = 784                              # number of input variables

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

D = D[0].astype('float32'), D[1].astype('float32')

training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(rng.randn(feats), name="w")

# initialize the bias term
b = theano.shared(0., name="b")


# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

X = theano.shared(D[0])
Y = theano.shared(D[1])

# Compile
train = theano.function(
          inputs=[],
          outputs=[],
          updates=[(w, w - 0.1 * gw), (b, b - 0.1 * gb)],
          givens=[(x,X),(y,Y)]
          )
'''
train = theano.function(
          inputs=[],
          outputs=[prediction, cost],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)),
          givens=[(x,X),(y,Y)]
          )
          
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, cost],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb))
          )
'''
predict = theano.function(inputs=[x], outputs=prediction)

# Train
print('Start')
s = time.time()
for i in range(training_steps):
	if i%100==0:
		print('Step :',i)
	train()
e = time.time()
print('Time elapsed : %.1fs'%(e-s))
print("Final model:")
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
