import sys  # isort:skip
import os  # isort:skip
sys.path.append(os.getcwd())  # isort:skip

import argparse
import os
import sys
from pathlib import Path

import numpy
from PIL import Image

from cppn.input_data import InputData, interp_z, sample_z
from cppn.model import CPPN, ModelConfig
from cppn.post_process_output import post_process_output
import chainer
import chainer.functions as F


def gen_z_circle(N):
    t = numpy.linspace(0, 2 * numpy.pi, N)
    x = numpy.cos(t)
    y = numpy.sin(t)
    return numpy.stack((x, y), axis=1).astype(numpy.float32)  # [N, 2]


def gen_mohei_movie():
    N = 30  # number of points
    zs = gen_z_circle(N)

    model_config = ModelConfig(width=1024, height=300, n_units_xyrz=128, n_hidden_units=[128, 128, 128, 128], z_size=2)
    model = CPPN(model_config)
    input_data = InputData(width=model_config.width, height=model_config.height, z_size=model_config.z_size)

    batch_size = 10
    images = []
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N - 1)
        print(f'{i} -> {j}')
        x, z = [], []
        for _z in zs[i:j]:
            _x, _z = input_data.as_batch(z=_z)
            x.append(_x)
            z.append(_z)

        if len(x) == 0: break

        x = numpy.concatenate(x)
        z = numpy.concatenate(z)

        # to device
        xp = model.xp
        x = chainer.Variable(xp.asarray(x))
        z = chainer.Variable(xp.asarray(z))

        y = model.forward(x, z)
        y = chainer.cuda.to_cpu(y.data)

        # chainer variable [B, 1, W, H], float [0, 1] -> numpy array uint8 [0, 255]
        y = post_process_output(y)

        for _y in y:
            images.append(Image.fromarray(_y[0]))

    # save as git
    images[0].save('./tmp/out.gif', save_all=True, append_images=images)


if __name__ == '__main__':
    gen_mohei_movie()