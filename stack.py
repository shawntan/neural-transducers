import theano
import theano.tensor as T
import numpy         as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

def cumsum(sequence):
    sumseq,_ = theano.scan(
            lambda x,csum: x+csum,
            sequences = [sequence],
            outputs_info = [np.float32(0)]
        )
    return sumseq


def build(size):
    def init(sequence_length):
        initial_V = T.alloc(np.float32(0),(sequence_length,size))
        initial_s = T.alloc(np.float32(0),(sequence_length,))
        def step(t,v,d,u,prev_V,prev_s):
            prev_V = prev_V[:t]
            V = T.concatenate([prev_V,v,initial_V[t+1:]])

            rev_cum_prev_s = cumsum(prev_s[:t][::-1])[::-1]
            to_flip_ = u - rev_cum_prev_s
            to_flip = (to_flip_ > 0) * to_flip_
            new_s_ = prev_s - to_flip
            new_s = (new_s_ > 0) * new_s_
            s = T.concatenate([new_s,d,initial_s[t+1:]])

            rev_cum_s = cumsum(new_s[::-1])[::-1] + d
            flip_score_ = 1 - rev_cum_s
            flip_score = (flip_score_ > 0) * flip_score_
            score = T.min([new_s,flip_score],axis=0)

            r = T.dot(score,prev_V) + d * v

            return V,s,r
