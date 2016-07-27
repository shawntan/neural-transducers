import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters


def rev_cumsum(seq):
    """
    cumsum,_ = theano.scan(
            lambda x,acc: acc + x,
            sequences=seq,
            outputs_info=[np.float32(0.)],
            go_backwards=True
        )
    return cumsum[::-1]
    """
    return T.cumsum(seq[::-1])[::-1]
def rectify(x):
    return (x > 0) * x
def build(size):
    def init(sequence_length):
        initial_V = T.alloc(np.float32(0), sequence_length, size)
        initial_s = T.alloc(np.float32(0), sequence_length)

        def step(t, v, d, u, prev_V, prev_s):
            prev_V_to_t = prev_V[:t]
            prev_s_to_t = prev_s[:t]
            V = T.concatenate([
                prev_V_to_t,
                v.dimshuffle('x', 0),
                initial_V[t + 1:]
            ])

            to_flip = rectify(u - rev_cumsum(prev_s[1:t+1]))
            new_s = rectify(prev_s_to_t - to_flip)

            s = T.concatenate([
                new_s,
                d.dimshuffle('x'),
                initial_s[t + 1:]
            ])

            flip_score = rectify(1 - rev_cumsum(s[1:t+1]))
            score = T.min([new_s, flip_score], axis=0)

            r = T.dot(score, prev_V_to_t) + d * v

            return V, s, r
        return initial_V, initial_s, step
    return init


if __name__ == "__main__":
    #print T.concatenate([
    #            rev_cumsum(T.arange(10)[1:]),
    #            [np.float32(0.)]
    #        ]).eval()
    stack_init = build(5)
    initial_V, initial_s, step = stack_init(10)
    V,s = initial_V,initial_s

    for t,(push,pop) in enumerate([ (1,0),(1,0),(0,1),(1,0),(1,0),(0,1),(0,1),(1,0)]):
        V, s, r = step(
            t=t,
            v=theano.shared(np.random.randn(5).astype(np.float32)),
            d=theano.shared(np.float32(push * 0.99)),
            u=theano.shared(np.float32(pop * 0.99)),
            prev_V=V,
            prev_s=s
        )

    f = theano.function(inputs=[], outputs=[V, s, r])
    V, s, r = f()

    print V
    print s
    print r
