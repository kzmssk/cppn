""" Genarate GIF movie of latent space linear interpolation with untrained model """
import argparse
import os
import sys
from pathlib import Path

import numpy
from PIL import Image

from cppn.input_data import InputData, interp_z, sample_z
from cppn.model import CPPN, ModelConfig
from cppn.post_process_output import post_process_output

# import CPPN modules
sys.path.append(os.getcwd())


def interp_movie():
    parser = argparse.ArgumentParser(description="Gen gif movie")
    parser.add_argument('--out', type=Path, default=Path('./tmp/out.gif'))
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--model_config_path', type=Path, default=Path('./conf/model.yaml'))
    args = parser.parse_args()

    # create directory to put result
    args.out.parent.mkdir(exist_ok=True)

    # init model
    model_config = ModelConfig.load(args.model_config_path)
    model = CPPN(model_config)

    # init x and z
    input_data = InputData(width=model_config.width, height=model_config.height, z_size=model_config.z_size)

    # gen frames
    images = []
    zs = interp_z(sample_z(model_config.z_size), sample_z(model_config.z_size), args.frames)
    for i in range(0, args.frames, args.batch_size):

        begin_idx = i
        end_idx = min(i + args.batch_size, args.frames - 1)
        print(f"{begin_idx} -> {end_idx}")

        # make input batch
        x = []
        z = []
        for _z in zs[begin_idx:end_idx]:
            _x, _z = input_data.as_batch(z=_z)
            x.append(_x)
            z.append(_z)

        x = numpy.concatenate(x)
        z = numpy.concatenate(z)

        y = model.forward(x, z)

        # chainer variable [B, 1, W, H], float [0, 1] -> numpy array uint8 [0, 255]
        y = post_process_output(y.data)

        for _y in y:
            images.append(Image.fromarray(_y[0]))

    # save as gif
    images[0].save(str(args.out), save_all=True, append_images=images)


if __name__ == '__main__':
    interp_movie()
