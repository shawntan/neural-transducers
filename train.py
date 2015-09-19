import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit import updates 
from theano_toolkit.parameters import Parameters
from theano_toolkit import hinton
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
    X = T.bvector('X')
    Y = T.ivector('Y')
    aux = {}

    predict = model.build(
        P,
        input_size=128,
        embedding_size=64,
        controller_size=256,
        stack_size=256,
        output_size=128,
    )

    output = predict(X,aux=aux)
    error = - T.log(output[T.arange(Y.shape[0]),((128+1 + Y)%(128+1))])
    error = error[-(Y.shape[0]/2):]
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
            updates=updates.adadelta(parameters,avg_grads,learning_rate=1e-8) + acc_clear
        )

    test = theano.function(
            inputs=[X],
            outputs=T.argmax(output,axis=1)[-(X.shape[0]/2):],
        )
    return acc,update,test

if __name__ == "__main__":
    acc,update,test = make_train_functions()
    import tasks
    error = np.inf
    count = 0
    while error > 0.01:
        length = np.random.randint(64 - 8) + 8
        total_error = 0
        total = 0
        for _ in xrange(10):
            x,y = tasks.reverse(128,length)
#            print x
#            print (129 + y)%129
            total_error += acc(x,y)
            total += 1
        error = total_error / total
        print error 
        update()
        count += 1
        if count % 20 == 0:
            x,y = tasks.reverse(128,10)
            print y[-(y.shape[0]/2):]
            print test(x)
