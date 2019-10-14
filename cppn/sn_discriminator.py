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
    def __init__(self, bottom_width=8, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(SNDiscriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = SNConvolution2D(1, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = SNConvolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = SNConvolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = SNConvolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = SNConvolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = SNConvolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = SNConvolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = SNLinear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def forward(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.c0_1(h))
        h = F.leaky_relu(self.c1_0(h))
        h = F.leaky_relu(self.c1_1(h))
        h = F.leaky_relu(self.c2_0(h))
        h = F.leaky_relu(self.c2_1(h))
        h = F.leaky_relu(self.c3_0(h))
        return self.l4(h)
