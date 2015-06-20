import theano
import theano.tensor as T
import numpy         as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

def cumsum(sequence):
    return theano.scan(
            lambda x,csum: x+csum,
            sequences = [sequence],
            output_info = [0]
        )

def build(size):
    def init(sequence_length):
        initial_V = T.alloc(np.float32(0),(sequence_length,size))
        initial_s = T.alloc(np.float32(0),(sequence_length,))
        def step(t,prev_V,prev_s,v,d,u):
            V = T.concatenate([prev_V[:t],v,initial_V[t+1:]])
            






if __name__ == "__main__":
    print cumsum(np.arange(10)).eval()
