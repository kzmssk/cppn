import typing
import chainer
import chainer.functions as F
import chainer.links as L
import dataclasses
import numpy

def post_process(x):
    """ post process of output by forward """
    if isinstance(x, chainer.Variable):
        x = x.data
    return (x * 255).astype(numpy.uint8)


@dataclasses.dataclass
class ModelConfig:
    width: int
    height: int
    n_units_xyrz: int
    n_hidden_units: typing.List[int]
    z_size: int
    activation: typing.Callable = F.tanh


class CPPN(chainer.Chain):
    def __init__(self, config: ModelConfig):
        self.config = config
        super(CPPN, self).__init__()

        initialW = chainer.initializers.Normal(scale=1.0)

        with self.init_scope():
            self.l_x = L.Linear(1, self.config.n_units_xyrz, initialW=initialW)
            self.l_y = L.Linear(1, self.config.n_units_xyrz, initialW=initialW)
            self.l_r = L.Linear(1, self.config.n_units_xyrz, initialW=initialW)
            self.l_z = L.Linear(self.config.z_size, self.config.n_units_xyrz, initialW=initialW)

            for i, n_hidden_unit in enumerate(self.config.n_hidden_units):
                setattr(self, f"l_hidden_{i}", L.Linear(None, n_hidden_unit, initialW=initialW))
            self.l_out = L.Linear(None, 1, initialW=initialW)

    def forward(self, x, z):
        f = self.config.activation
        _x, _y, _r = F.split_axis(x, 3, axis=1)
        h = self.l_x(_x)
        h += self.l_y(_y)
        h += self.l_r(_r)
        h += self.l_z(z)
        h = f(h)

        for i in range(len(self.config.n_hidden_units)):
            h = f(getattr(self, f"l_hidden_{i}")(h))
        h = F.sigmoid(self.l_out(h))
        h = h.reshape((1, self.config.width, self.config.height))
        return h
