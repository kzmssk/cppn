# import chainer gan lib
import sys  # isort:skip
from pathlib import Path  # isort:skip
CHAINER_GAN_LIB = Path(__file__).parent.parent / 'chainer-gan-lib'  # isort:skip
sys.path.append(str(CHAINER_GAN_LIB))  # isort:skip
sys.path.append(str(CHAINER_GAN_LIB / 'sn'))  # isort:skip
sys.path.append(str(CHAINER_GAN_LIB / 'common'))  # isort:skip

import chainer
import chainer.functions as F

from sn.sn_convolution_2d import SNConvolution2D
from sn.sn_linear import SNLinear


class SNDiscriminator(chainer.Chain):
    def __init__(self, wscale=0.02, ch=6):
        w = chainer.initializers.Normal(wscale)
        super(SNDiscriminator, self).__init__()
        with self.init_scope():
            self.c_1 = SNConvolution2D(1, ch, 4, 2, 1, initialW=w)
            self.c_2 = SNConvolution2D(ch, ch * 2, 4, 2, 1, initialW=w)
            self.c_3 = SNConvolution2D(ch * 2, ch * 4, 3, 2, 1, initialW=w)
            self.c_4 = SNConvolution2D(ch * 4, ch * 8, 3, 2, 1, initialW=w)
            self.l_5 = SNLinear(None, 1, initialW=w)

    def forward(self, x):
        h = F.leaky_relu(self.c_1(x))
        h = F.leaky_relu(self.c_2(h))
        h = F.leaky_relu(self.c_3(h))
        h = F.leaky_relu(self.c_4(h))
        return self.l_5(h)