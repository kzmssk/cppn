""" Genarate GIF movie of latent space linear interpolation with untrained model """
from os.path import exists
import sys
import os
import argparse
from pathlib import Path
from PIL import Image

# import CPPN modules
sys.path.append(os.getcwd())
from cppn.input_data import InputData, interp_z, sample_z
from cppn.model import CPPN, ModelConfig, post_process


def interp_movie():
    parser = argparse.ArgumentParser(description="Gen gif movie")
    parser.add_argument('--out', type=Path, default=Path('./tmp/out.gif'))
    parser.add_argument('--frames', type=int, default=20)
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
    for z in interp_z(sample_z(model_config.z_size), sample_z(model_config.z_size), args.frames):
        x, z = input_data.as_batch(z=z)
        y = post_process(model.forward(x, z))[0]
        images.append(Image.fromarray(y))

    # save as gif
    images[0].save(str(args.out), save_all=True, append_images=images)


if __name__ == '__main__':
    interp_movie()
