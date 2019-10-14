import chainer
import chainer.functions as F
import numpy

from cppn.conditional_model import ConditionalCPPN, ConditionalModelConfig
from cppn.input_data import InputData
from cppn.model import CPPN, ModelConfig


def get_dammy_input(batch, width, height, channel):
    x = numpy.random.rand(batch, channel, width, height)  # [0, 1]
    x *= 255
    return x.astype(numpy.float32)


def get_dammy_output(batch, width, height):
    x = numpy.random.rand(batch, 1, width, height)  # [0, 1]
    x *= 255
    return x.astype(numpy.float32)


def gen_input_batch(batch_size, width, height, z_size):
    # create inputs
    inputs = {}
    x, z = [], []
    for idx in range(batch_size):
        _x, _z = InputData(width=width, height=height, z_size=z_size).as_batch()
        _x = chainer.Variable(_x)
        _z = chainer.Variable(_z)
        x.append(_x)
        z.append(_z)
        inputs[idx] = (_x, _z)
    x = F.concat(x, axis=0)
    z = F.concat(z, axis=0)
    return x, z, inputs


def test_unconditional_forward():
    """ checking gradient leaking along batch axis """
    width = 5
    height = 7
    z_size = 2
    batch_size = 3

    model = CPPN(ModelConfig(width=width, height=height, n_units_xyrz=3, n_hidden_units=[5, 5], z_size=z_size))
    model.zerograds()

    # create inputs: inputs is dict whose key is batch index, and value is tuple of (x, z) for each index
    x, z, inputs = gen_input_batch(batch_size, width, height, z_size)

    # forward prop
    y = model.forward(x, z)

    # taking loss at only first image
    t = get_dammy_output(batch_size, width, height)
    loss = F.mean_squared_error(y[0], t[0])

    # check gradient leaking
    assert sum([g.data.sum() for g in chainer.grad((loss, ), inputs[0])]) != 0.0
    assert sum([g.data.sum() for g in chainer.grad((loss, ), inputs[1])]) == 0.0
    assert sum([g.data.sum() for g in chainer.grad((loss, ), inputs[2])]) == 0.0


def test_conditional_backward():
    """ checking gradient leaking along batch axis """
    width = 5
    height = 7
    z_size = 2
    batch_size = 3

    model = ConditionalCPPN(
        ConditionalModelConfig(width=width,
                               height=height,
                               n_units_xyr=3,
                               n_hidden_units=[
                                   10,
                                   10,
                               ],
                               z_size=z_size,
                               in_width=64,
                               in_height=64,
                               in_channel=1,
                               use_batch_norm=False))
    model.zerograds()

    # create inputs: inputs is dict whose key is batch index, and value is tuple of (x, z) for each index
    x, z, inputs = gen_input_batch(batch_size, width, height, z_size)
    c = chainer.Variable(get_dammy_input(batch_size, 64, 64, 1))  # init dammy conditional input

    # forward prop
    y = model.forward(x, z, c)

    # taking loss at only first image
    t = get_dammy_output(batch_size, width, height)
    loss = F.mean_squared_error(y[0], t[0])

    g_x, g_z = chainer.grad((loss, ), inputs[0])
    g_c = chainer.grad((loss, ), (c, ))[0].data

    assert g_c[0].sum() != 0.0, f"gradient of c is zero"
    assert g_x.data.sum() != 0.0, f"gradient of x is zero"
    assert g_z.data.sum() != 0.0, f"gradient of z is zero"

    g_x, g_z = chainer.grad((loss, ), inputs[1])
    assert g_c[1].sum() == 0.0, f"gradient of c is zero"
    assert g_x.data.sum() == 0.0, f"gradient of x is zero"
    assert g_z.data.sum() == 0.0, f"gradient of z is zero"

    g_x, g_z = chainer.grad((loss, ), inputs[2])
    assert g_c[2].sum() == 0.0, f"gradient of c is zero"
    assert g_x.data.sum() == 0.0, f"gradient of x is zero"
    assert g_z.data.sum() == 0.0, f"gradient of z is zero"
