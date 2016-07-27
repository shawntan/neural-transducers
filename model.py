import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

import lstm
import stack
print __name__

def build(P, input_size, embedding_size, controller_size, stack_size, output_size):
    softmax_output_size = output_size + 1
    P.embeddings = np.random.randn(input_size + 2,embedding_size).astype(np.float32)
    controller_step = lstm.build_step(
        P, name="controller",
        input_size = embedding_size + stack_size,
        hidden_size = controller_size
    )
    stack_init = stack.build(size=stack_size)

    P.W_controller_output = 0.1 * np.random.randn(
        controller_size,
        softmax_output_size + stack_size + 1 + 1
    ).astype(np.float32)
    bias = np.zeros((softmax_output_size + stack_size + 1 + 1,), dtype=np.float32)
    bias[-2] = 5
    bias[-1] = -5
    P.b_controller_output = bias

    init_controller_cell   = np.zeros((controller_size,), dtype=np.float32)
    init_controller_hidden = np.zeros((controller_size,), dtype=np.float32)
    init_stack_r           = np.zeros((stack_size,), dtype=np.float32)

    def predict(ids,aux={}):
        X = P.embeddings[ids]
        init_stack_V, init_stack_s, stack_step = stack_init(X.shape[0])

        def step(x, t,
                 prev_controller_cell, prev_controller_hidden,
                 prev_V, prev_s, prev_r):

            controller_input = T.concatenate([x, prev_r])
            controller_cell, controller_hidden = \
                controller_step(
                    x=controller_input,
                    prev_cell=prev_controller_cell,
                    prev_hidden=prev_controller_hidden
                )

            controller_output = T.dot(controller_hidden, P.W_controller_output) +\
                P.b_controller_output

            output = controller_output[:softmax_output_size]
            v      = T.tanh(controller_output[
                        softmax_output_size:
                        softmax_output_size + stack_size
                    ])
            flags = T.nnet.sigmoid(controller_output[-2:])

            V, s, r = stack_step(
                t=t,
                v=v,
                d=flags[0],
                u=flags[1],
                prev_V=prev_V, prev_s=prev_s
            )

            return controller_cell, controller_hidden, V, s, r, controller_output, output

        sequences, _ = theano.scan(
            step,
            sequences=[X, T.arange(X.shape[0])],
            outputs_info=[
                init_controller_cell,
                init_controller_hidden,
                init_stack_V,
                init_stack_s,
                init_stack_r,
                None,
                None
            ]
        )

        outputs = T.nnet.softmax(sequences[-1])
        aux['controller_output'] = sequences[-2]

        return outputs
    return predict

if __name__ == "__main__":
    P = Parameters()
    X = T.matrix('X')
    predict = build(P,
                    input_size=5,
                    controller_size=5,
                    stack_size=5,
                    output_size=5
                    )
    f = theano.function(
        inputs=[X],
        outputs=predict(X)
    )
    print f(np.random.randn(10, 5).astype(np.float32))
