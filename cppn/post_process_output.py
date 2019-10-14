import chainer
import numpy


def post_process_output(x):
    """ convert output of CPPN -> uint8 array """
    if isinstance(x, chainer.Variable):
        x = x.data

    return numpy.clip(x * 127.5 + 127.5, 0.0, 255.0).astype(numpy.uint8)
