import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit import updates 
from theano_toolkit.parameters import Parameters
from theano_toolkit import hinton
from theano_toolkit.cache import cache
import model

def clip(delta,thresh):
    thresh = np.float32(thresh)
    norm = T.sqrt(T.sum(delta**2))
    return T.switch(
            T.gt(norm,thresh),
            thresh * delta/norm,
            delta
        )

def make_train_functions():
    P = Parameters()
    X = T.bmatrix('X')
    Y = T.bmatrix('Y')
    aux = {}
    predict = model.build(
        P,
        input_size=8,
        controller_size=256,
        stack_size=256,
        output_size=8,
    )

    output = T.nnet.sigmoid(predict(X,aux=aux))
    error = - T.sum(Y * T.log(output) + (1-Y) * T.log(1-output),axis=1)
    parameters = P.values()
    gradients = T.grad(T.sum(error),wrt=parameters)
    shapes = [ p.get_value().shape for p in parameters ]
    count = theano.shared(np.float32(0))
    acc_grads  = [
        theano.shared(np.zeros(s,dtype=np.float32))
        for s in shapes
    ]

    acc_update = [ (a,a+g) for a,g in zip(acc_grads,gradients) ] +\
                 [ (count,count + np.float32(1)) ]
    acc_clear = [ (a,np.float32(0) * a) for a in acc_grads ] +\
                [ (count,np.int32(0)) ]
    avg_grads = [ (g / count) for g in acc_grads ]
    avg_grads = [ clip(g,1) for g in acc_grads ]


    acc = theano.function(
            inputs=[X,Y],
            outputs=T.mean(error),
            updates = acc_update,
        )
    update = theano.function(
            inputs=[],
            updates=updates.rmsprop(parameters,avg_grads) + acc_clear
        )

    test = theano.function(
            inputs=[X],
            outputs=T.nnet.sigmoid(aux['controller_output']),
        )
    return acc,update,test

if __name__ == "__main__":
    acc,update,test = make_train_functions()
    import tasks
    error = np.inf
    while error > 0.01:
        length = np.random.randint(64 - 8) + 8
        for _ in xrange(10):
            x,y = tasks.copy(7,length)
            error = acc(x,y)
            print error
        update()
        print

    x,y = tasks.copy(7,20)
    print hinton.plot(test(x))


