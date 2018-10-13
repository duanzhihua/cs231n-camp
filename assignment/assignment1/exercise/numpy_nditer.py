# -*- coding: utf-8 -*-

import numpy as np
def iter_add_py(x, y, out=None):
    addop = np.add
    it = np.nditer([x, y, out], [],
                [['readonly'], ['readonly'], ['writeonly','allocate']])
    with it:
        for (a, b, c) in it:
            addop(a, b, out=c)
    return it.operands[2]

def iter_add(x, y, out=None):
    addop = np.add

    it = np.nditer([x, y, out], [],
                [['readonly'], ['readonly'], ['writeonly','allocate']])
    with it:
        while not it.finished:
            addop(it[0], it[1], out=it[2])
            it.iternext()

        return it.operands[2]
    
    
def outer_it(x, y, out=None):
    mulop = np.multiply

    it = np.nditer([x, y, out], ['external_loop'],
            [['readonly'], ['readonly'], ['writeonly', 'allocate']],
            op_axes=[list(range(x.ndim)) + [-1] * y.ndim,
                     [-1] * x.ndim + list(range(y.ndim)),
                     None])   
    for (a, b, c) in it:
            mulop(a, b, out=c)
    return it.operands[2]


a = np.arange(2)+1
b = np.arange(3)+1
outer_it(a,b)




def luf(lamdaexpr, *args, **kwargs):
    "luf(lambdaexpr, op1, ..., opn, out=None, order='K', casting='safe', buffersize=0)"
    nargs = len(args)
    op = (kwargs.get('out',None),) + args
    it = np.nditer(op, ['buffered','external_loop'],
            [['writeonly','allocate','no_broadcast']] +
                            [['readonly','nbo','aligned']]*nargs,
            order=kwargs.get('order','K'),
            casting=kwargs.get('casting','safe'),
            buffersize=kwargs.get('buffersize',0))
    while not it.finished:
        it[0] = lamdaexpr(*it[1:])
        it.iternext()
        return it.operands[0]

a = np.arange(5)
b = np.ones(5)
luf(lambda i,j:i*i + j/2, a, b)



a = np.arange(6, dtype='i4')[::-2]
with nditer(a, [],
        [['writeonly', 'updateifcopy']],
        casting='unsafe',
        op_dtypes=[np.dtype('f4')]) as i:
    x = i.operands[0]
    x[:] = [-1, -2, -3]
    # a still unchanged here
a, x







