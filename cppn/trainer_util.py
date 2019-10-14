import subprocess
from pathlib import Path

import chainer
import numpy
import yaml
from PIL import Image

from cppn.post_process_output import post_process_output


def sample_generate(generator, save_dir_path, input_data, rows=5, cols=5, seed=0):
    """ Perform rows*cols images random generation """
    @chainer.training.make_extension()
    def make_image(trainer):
        numpy.random.seed(seed)
        xp = generator.xp

        N = rows * cols  # number of images

        # make x and z
        x, z = [], []
        for _ in range(N):
            _x, _z = input_data.as_batch()
            x.append(_x)
            z.append(_z)

        x = numpy.concatenate(x)
        z = numpy.concatenate(z)

        x = chainer.Variable(xp.asarray(x))
        z = chainer.Variable(xp.asarray(z))

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = generator.forward(x, z)

        x = chainer.cuda.to_cpu(x.data)
        numpy.random.seed()

        # float -> uint8
        x = post_process_output(x)

        x = x.reshape((rows, cols, 1, input_data.height, input_data.width))
        x = x.reshape(
            (rows * input_data.height, cols * input_data.width))  # output is gray scale so that image array is 2dim

        preview_dir = save_dir_path / 'preview'
        preview_dir.mkdir(exist_ok=True, parents=False)

        save_path = preview_dir / 'image{:0>8}.png'.format(trainer.updater.iteration)
        Image.fromarray(x).save(save_path)

    return make_image


def snap_exec_info(log_dir_path, argparse_args):
    """ Save execution info to log_dir_path """

    # commandline args
    def process_val(val):
        if isinstance(val, Path):
            return str(val)
        else:
            return val

    argparse_args = {key: process_val(val) for key, val in vars(argparse_args).items()}
    with open(log_dir_path / 'args.yaml', 'w') as f:
        yaml.dump(argparse_args, f)

    # git status
    def save_cmd_output(save_path, cmd):
        with open(save_path, 'wb') as f:
            f.write(subprocess.check_output(cmd.split()))

    save_cmd_output(log_dir_path / "git-head.txt", "git rev-parse HEAD")
    save_cmd_output(log_dir_path / "git-status.txt", "git status")
    save_cmd_output(log_dir_path / "git-log.txt", "git log")
    save_cmd_output(log_dir_path / "git-diff.txt", "git diff")
