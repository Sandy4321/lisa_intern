import theano
import pylearn2
import numpy as np
from pylearn2.utils import sharedX
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import config
def project(w, x):
        """
        Takes a sequence of integers and projects (embeds) these labels
        into a continuous space by concatenating the correspending
        rows in the projection matrix W i.e. [2, 5] -> [W[2] ... W[5]]

        Parameters
        ----------
        x : theano.tensor, int dtype
            A vector of labels (or a matrix where each row is a sample in
            a batch) which will be projected
        """

        #assert 'int' in x.dtype
        #print x.ndim

        if x.ndim == 2:
            shape = (x.shape[0], x.shape[1] * w.shape[1])
            return w[x.flatten()].reshape(shape)
        elif x.ndim == 1:
            return w[x].flatten()
        else:
            assert ValueError("project needs 1- or 2-dimensional input")

def context( state_below):
    "q^(h) from EQ. 2"

    state_below = state_below.reshape((state_below.shape[0], dim, 6))
    rval = C.dimshuffle('x', 0, 1) * state_below
    #print 'in context'
    #print C.dimshuffle('x',0,1).eval().shape
    #print state_below.eval().shape
    rval = rval.sum(axis=2)
    #print rval.eval().shape
    return rval

def score(X, Y=None, k = 1):
    X = project(W,X)
    q_h = context(X)
    #print 'in score'
    # this is used during training
    if Y is not None:
        if Y.ndim != 1:
            Y = Y.flatten().dimshuffle(0)
            #print 'reshaped'
        q_w = project(W,Y)
        
        #print q_h.eval().shape
        #print q_w.eval().shape
        #print q_w.reshape((k, X.shape[0], q_h.shape[1])).eval().shape
        
        rval = (q_w.reshape((k, X.shape[0], q_h.shape[1])) * q_h).sum(axis=2)
        rval = rval + B[Y].reshape((k, X.shape[0]))
    # during nll
    else:
        q_w = W
        rval = T.dot(q_h, q_w.T) + B.dimshuffle('x', 0)
    return rval

def delta(data, k = 1):

    X, Y = data
    if Y.ndim != 1:
        Y = Y.flatten().dimshuffle(0)

    #if noise_p is None:
    p_n = 1. / v
    rval = score(X, Y,k) - T.log(k * p_n)
    # else:
    #     p_n = noise_p
    #     rval = score(X, Y, k = k)
    #     rval = rval - T.log(k * p_n[Y]).reshape(rval.shape)
    return T.cast(rval, config.floatX)

v = 10000.
#vocab size
k = 15
dim = 5

n = 10
#num examples

X = theano.shared(np.random.randint(0,v,size=(n,6)))
#x.shape = (15,6)
Y = theano.shared(np.random.randint(0,v,size=(n,1)))
#y.shape = (15,1)
W = sharedX(np.random.rand(v,dim))
#w.shape = 10,5
B = sharedX(np.random.rand(v,1))
#b.shape = 10,1
c = np.random.randint(0,100,(dim,6))
C = sharedX(c)
#c shape = 5,6
#we get a column of 5 dimensions for each context word



# #rproj = w[x.flatten()]
# #rproj.shape
# ##90,5
# #shape = (x.shape[0], x.shape[1] * w.shape[1])
# #rproj = rproj.reshape(shape)
# #15 examples of 6 words each and 5 dim for each word
# #so 15 rows. where each row has 6*5 dim
# #sb = rproj.reshape(rproj.shape[0],k,6)
# #sb.shape
# #made it 15,5,6 now

# #c shape is 1,5,6
# #so that for each example we still use the same C columns
# qh = (C.dimshuffle('x',0,1)*sharedX(sb)).sum(axis=2)

# ally = np.arange(v).reshape(v,1)

# qw = project(w,y)
# allqw = project(w,ally)

# swh = (qw*qh).sum(axis=1) + b[y].flatten()
# sallwh = theano.tensor.dot(qh,allqw.T)+b[ally].flatten()

# soft = theano.tensor.nnet.softmax(sallwh)
# probsoft = T.diag(soft[(T.arange(y.shape[0]),y)])

# esallwh = T.exp(sallwh)
# eswh = T.exp(swh)
# esallwh = esallwh.sum(axis=1)

# prob = eswh/esallwh

#done


data = (X,Y)
params = [W[Y.flatten()],B[Y.flatten()],C]
# for p in params:
#      p = p[Y]
sc = score(X,Y)
print 'scores X,Y shape is'
print sc.eval().shape
pos_ = T.jacobian(sc.flatten(),params,disconnected_inputs='ignore')
pos_coeff = 1 - T.nnet.sigmoid(delta(data)).flatten()

pos = []
for param in pos_:
     axes = [0]
     axes.extend(['x' for item in range(param.ndim-1)])
     pos.append(pos_coeff.dimshuffle(axes)*param)
del pos_,pos_coeff

noise = np.ones((n*k),dtype='int32')

sc = score(X,noise,k)
print 'scores X,noise,k'
print sc.eval().shape
neg_ = T.jacobian(sc.flatten(), params, disconnected_inputs='ignore')
neg_coeff = T.nnet.sigmoid(delta((X, noise),k))
neg = []
for param in neg_:
    axes = [0,1]
    axes.extend(['x' for item in range(param.ndim - 1)])
#     #print param.shape.eval()
    newshape = (neg_coeff.shape[0],neg_coeff.shape[1],param.shape[1],param.shape[2])
    tmp = neg_coeff.dimshuffle(axes) * param.reshape(newshape)
    tmp = tmp.sum(axis=0)
    neg.append(tmp)
del neg_, neg_coeff