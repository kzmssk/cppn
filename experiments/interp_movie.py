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
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('./tmp/out.gif'))
    parser.add_argument('--frames', type=int, default=20)
    args = parser.parse_args()
    
    # model parameters
    width = 128
    height = 128
    z_size = 2
    h_size = 25

    # init model
    model = CPPN(ModelConfig(
        width=width,
        height=height,
        n_units_xyrz=h_size,
        n_hidden_units=[h_size, h_size, h_size],
        z_size=z_size
    ))

    # init x and z
    input_data = InputData(width=width, height=height, z_size=z_size)

    # gen frames
    images = []
    for z in interp_z(sample_z(z_size), sample_z(z_size), args.frames):
        x, z = input_data.as_batch(z=z)
        y = post_process(model.forward(x, z))[0]
        images.append(Image.fromarray(y))

    # save as gif
    images[0].save(str(args.out), save_all=True, append_images=images)

if __name__ == '__main__':
    interp_movie() 
