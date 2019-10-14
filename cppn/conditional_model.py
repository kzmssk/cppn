import dataclasses
import typing

import chainer
import chainer.functions as F
import chainer.links as L

from cppn import config_base


@dataclasses.dataclass
class ConditionalModelConfig(config_base.ConfigBase):
    width: int
    height: int
    n_units_xyr: int
    n_hidden_units: typing.List[int]
    z_size: int
    in_width: int = 64  # width of conditional input
    in_height: int = 64  # height of conditional input
    in_channel: int = 1  # channel of conditional input
    activation: typing.Callable = F.tanh
    use_batch_norm: bool = True  # using batch normalization layer or not


class Block(chainer.Chain):
    def __init__(self, in_channel, out_channel, ksize, activation, use_batch_norm):
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channel, out_channel, ksize=ksize, stride=2)
            if use_batch_norm:
                self.norm = L.BatchNormalization(out_channel)

    def __call__(self, x):
        h = self.conv(x)
        if self.use_batch_norm: h = self.norm(h)
        return self.activation(h)


class ConditionInputProcessor(chainer.Chain):
    """ Conditional input processing part of CPPN model """
    def __init__(self, width, height, channel, activation, use_batch_norm):
        self.width = width
        self.height = height
        self.channel = channel
        super(ConditionInputProcessor, self).__init__()

        with self.init_scope():
            in_channel = channel
            for i, out_channel, ksize in zip(range(3), (12, 32, 64, 64), (6, 3, 3, 2)):
                setattr(self, f"block_{i}",
                        Block(in_channel, out_channel, ksize, activation, use_batch_norm=use_batch_norm))
                in_channel = out_channel

    def __call__(self, x):
        batch_size = x.shape[0]
        h = x
        for i in range(3):
            block = getattr(self, f"block_{i}")
            h = block(h)
        h = F.average_pooling_2d(h, ksize=4, stride=2)
        return F.reshape(h, (batch_size, -1))  # reshape into vector


class ConditionalCPPN(chainer.Chain):
    """ conditional generator of CPPN """
    def __init__(self, config: ConditionalModelConfig):
        self.config = config
        super(ConditionalCPPN, self).__init__()

        initialW = chainer.initializers.Normal(scale=1.0)

        with self.init_scope():
            self.l_x = L.Linear(1, self.config.n_units_xyr, initialW=initialW)
            self.l_y = L.Linear(1, self.config.n_units_xyr, initialW=initialW)
            self.l_r = L.Linear(1, self.config.n_units_xyr, initialW=initialW)
            self.l_z = L.Linear(self.config.z_size, self.config.n_units_xyr, initialW=initialW)
            self.l_c = ConditionInputProcessor(self.config.in_width, self.config.in_height, self.config.in_channel,
                                               self.config.activation, self.config.use_batch_norm)

            for i, n_hidden_unit in enumerate(self.config.n_hidden_units):
                setattr(self, f"l_hidden_{i}", L.Linear(None, n_hidden_unit, initialW=initialW))
            self.l_out = L.Linear(None, 1, initialW=initialW)

    def forward(self, x, z, c):
        assert x.shape[0] % (self.config.width *
                             self.config.height) == 0, f"Invalid input size x.shape[0] % (width * height) != 0"

        batch_size = x.shape[0] // (self.config.width * self.config.height)

        # processing conditional input
        h_c = self.l_c(c)  # [B, 256]
        h_c = [
            F.repeat(_h, self.config.width * self.config.height, axis=0) for _h in F.split_axis(h_c, batch_size, axis=0)
        ]
        h_c = F.concat(h_c, axis=0)

        # process input
        f = self.config.activation
        _x, _y, _r = F.split_axis(x, 3, axis=1)
        h_in = self.l_x(_x)
        h_in += self.l_y(_y)
        h_in += self.l_r(_r)
        h_in += self.l_z(z)
        h_in = f(h_in)

        # concat with conditional feature
        h = F.concat((h_in, h_c), axis=1)

        for i in range(len(self.config.n_hidden_units)):
            h = f(getattr(self, f"l_hidden_{i}")(h))

        h = F.sigmoid(self.l_out(h))

        h = F.concat(
            [_h.reshape((1, 1, self.config.width, self.config.height)) for _h in F.split_axis(h, batch_size, axis=0)],
            axis=0)
        return h
