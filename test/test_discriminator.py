import chainer
import numpy

from cppn import sn_discriminator


def get_dammy_input(batch, width, height, channel):
    x = numpy.random.rand(batch, channel, width, height)  # [0, 1]
    x *= 255
    return x.astype(numpy.float32)


def test_forward():
    batch = 1
    width = 64
    height = 64
    channel = 1

    # init model
    discriminator = sn_discriminator.SNDiscriminator()

    # init dammy input
    x = chainer.Variable(get_dammy_input(batch, width, height, channel))

    # forward prop
    y = discriminator.forward(x)

    assert y.shape == (batch, 1)
