""" Train CPPN model with image dataset """
# import CPPN modules
import sys  # isort:skip
import os  # isort:skip
sys.path.append(os.getcwd())  # isort:skip

import typing

import argparse
import dataclasses
import shutil
from pathlib import Path

import chainer
from chainer import training
from chainer.training import extension, extensions

from cppn import (config_base, model, my_dataset, mse_updater, trainer_util)
from cppn.model import ModelConfig


@dataclasses.dataclass
class TrainConfig(config_base.ConfigBase):
    """ Training configuration """
    max_iter: int  # max number of training iteration
    batch_size: int
    snapshot_iter_interval: int
    display_iter_interval: int
    evaluation_iter_interval: int
    display_iter_interval: int
    train_image_dir_path: typing.Optional[Path] = None # root directory of image files
    train_image_path_txt: typing.Optional[Path] = None # path to text files of image file paths
    alpha: float = 0.0002  # alpha of adam optimizer
    beta1: float = 0.0  # beta1 of adam optimizer
    beta2: float = 0.9  # beta2 of adam optimizer


def init_optimizer(model, alpha, beta1, beta2):
    """ initialize optimizer with model """
    opt = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    opt.setup(model)
    return opt


def train_mse(log_dir_path: Path, train_config: TrainConfig, model_config: model.ModelConfig, device: int):
    """ Train CPPN model on image dataset """
    # init dataset

    if train_config.train_image_dir_path is not None:
        paths = list(train_config.train_image_dir_path.glob('*.jpg'))  # TODO: take also PNG images
    elif train_config.train_image_path_txt is not None:
        with open(train_config.train_image_path_txt) as f:
            paths = f.readlines()
        paths = [ p.split('\n')[0] for p in paths ]
    else:
        raise RuntimeError('train_image_dir_path or train_image_path_txt should be specified')
    
    train_dataset = my_dataset.MyDataset(paths, model_config.width, model_config.height, model_config.z_size)
    train_iterator = chainer.iterators.SerialIterator(train_dataset, train_config.batch_size)

    # init generator
    generator = model.CPPN(model_config)

    # copy model to device
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        generator.to_gpu()

    # init optimizers
    gen_opt = init_optimizer(generator, train_config.alpha, train_config.beta1, train_config.beta2)

    # init updater
    updater = mse_updater.MseUpdater(iterator=train_iterator,
                                   gen=generator,
                                   gen_opt=gen_opt,
                                   input_data=train_dataset.input_data,
                                   device=device)
    trainer = training.Trainer(updater, (train_config.max_iter, 'iteration'), out=log_dir_path)

    # --- init updater's hooks (logging)
    # snapshot of models
    trainer.extend(extensions.snapshot_object(generator, 'generator_{.updater.iteration}.npz'),
                   trigger=(train_config.snapshot_iter_interval, 'iteration'))

    # report log
    report_keys = ["loss_gen"]
    trainer.extend(extensions.LogReport(keys=report_keys, trigger=(train_config.display_iter_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(train_config.display_iter_interval, 'iteration'))
    trainer.extend(trainer_util.sample_generate(generator, log_dir_path, train_dataset.input_data),
                   trigger=(train_config.evaluation_iter_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # start training
    trainer.run()


def start_train_mse():
    parser = argparse.ArgumentParser(description="Training CPPN model with mean squared error")
    parser.add_argument('log_dir_path', type=Path)
    parser.add_argument('--train_config_path', type=Path, default='./conf/train_mse.yaml')
    parser.add_argument('--model_config_path', type=Path, default='./conf/model.yaml')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    # init log_dir
    args.log_dir_path.mkdir(parents=False, exist_ok=False)

    # load train config
    train_config = TrainConfig.load(args.train_config_path)

    # load model config
    model_config = ModelConfig.load(args.model_config_path)

    # store execution info to log_dir
    trainer_util.snap_exec_info(args.log_dir_path, args)

    # copy configs to log dir
    shutil.copyfile(args.train_config_path, args.log_dir_path / args.train_config_path.name)
    shutil.copyfile(args.model_config_path, args.log_dir_path / args.model_config_path.name)

    # start train
    train_mse(args.log_dir_path, train_config, model_config, args.device)


if __name__ == '__main__':
    start_train_mse()
