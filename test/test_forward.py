from cppn.model import CPPN, ModelConfig
from cppn.input_data import InputData
from cppn.conditional_model import ConditionalCPPN, ConditionalModelConfig
import numpy


def get_dammy_input(batch, width, height, channel):
    x = numpy.random.rand(batch, channel, width, height)  # [0, 1]
    x *= 255
    return x.astype(numpy.float32)


def test_unconditional_forward():
    width = 5
    height = 7
    z_size = 2
    batch_size = 3

    model = CPPN(ModelConfig(width=width, height=height, n_units_xyrz=3, n_hidden_units=[5, 5], z_size=z_size))

    x, z = [], []
    for _ in range(batch_size):
        _x, _z = InputData(width=width, height=height, z_size=z_size).as_batch()
        x.append(_x)
        z.append(_z)
    x = numpy.concatenate(x, axis=0)
    z = numpy.concatenate(z, axis=0)
    y = model.forward(x, z)
    assert y.shape == (batch_size, width, height)


def test_conditional_forward():
    batch_size = 3
    model = ConditionalCPPN(
        ConditionalModelConfig(width=12,
                               height=13,
                               n_units_xyr=3,
                               n_hidden_units=[
                                   10,
                                   10,
                               ],
                               z_size=2,
                               in_width=64,
                               in_height=64,
                               in_channel=1))
    x, z = [], []
    for _ in range(batch_size):
        _x, _z = InputData(width=12, height=13, z_size=2).as_batch()
        x.append(_x)
        z.append(_z)
    x = numpy.concatenate(x, axis=0)
    z = numpy.concatenate(z, axis=0)
    c = get_dammy_input(batch_size, 64, 64, 1)  # init dammy conditional input
    y = model.forward(x, z, c)
    assert y.shape == (batch_size, 12, 13)
