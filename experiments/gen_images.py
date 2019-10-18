# import CPPN modules
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


def gen_images():
    parser = argparse.ArgumentParser(description="multiple images as single image")
    parser.add_argument('--out', type=Path, default=Path('./tmp/out.png'))
    parser.add_argument('--n_rows', type=int, default=5)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--model_config_path', type=Path, default=Path('./conf/model.yaml'))
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--load', type=Path)
    parser.add_argument('--size', type=int)
    args = parser.parse_args()

    batch_size = args.n_rows * args.n_cols

    # init model
    model_config = ModelConfig.load(args.model_config_path)
    model = CPPN(model_config)

    # override size of output
    if args.size:
        model_config.width = args.size
        model_config.height = args.size

    if args.load:
        assert args.load.exists()
        print(f"load model from {args.load}")
        chainer.serializers.load_npz(args.load, model)

    # model to gpu
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # init x and z
    input_data = InputData(width=model_config.width, height=model_config.height, z_size=model_config.z_size)

    x, z = [], []
    for _ in range(batch_size):
        _x, _z = input_data.as_batch()
        x.append(_x)
        z.append(_z)

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
    y = y.reshape((args.n_rows, args.n_cols, 1, input_data.height, input_data.width))
    y = y.transpose((0, 3, 1, 4, 2))
    y = y.reshape((args.n_rows * input_data.height, args.n_cols * input_data.width))
    Image.fromarray(y).save(args.out)


if __name__ == '__main__':
    gen_images()