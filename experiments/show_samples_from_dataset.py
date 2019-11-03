# import CPPN modules
import sys  # isort:skip
import os  # isort:skip
sys.path.append(os.getcwd())  # isort:skip

import numpy
import argparse
from pathlib import Path
import random
from PIL import Image
from cppn.mnist_dataset import MnistDataset
from cppn.emnist_dataset import EMnistDataset


def show_samples_from_dataset(
    dataset_type: str,
    n_rows: int,
    n_cols: int,
    size: int,
    data_path: Path,
    output_path: Path):
    
    # initialize dataset
    if dataset_type == 'mnist':
        dataset = MnistDataset(width=size, height=size, z_size=1)
    elif dataset_type == 'emnist':
        dataset = EMnistDataset(width=size, height=size, z_size=1, data_path=data_path)
    else:
        raise NotImplementedError

    # sample images
    rows = []
    for i in range(n_rows):
        rows.append(
            numpy.concatenate([ dataset.get_example(random.randint(0, len(dataset) - 1))[2] for _ in range(n_cols)], axis=3)
        )
    img = numpy.concatenate(rows, axis=2)
    
    # float32 -> uint8
    img = (img[0, 0] * 255.0).astype(numpy.uint8)

    Image.fromarray(img).save(output_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--dataset_type', type=str, choices=('mnist', 'emnist'), default='mnist')
    parser.add_argument('--data_path', type=Path)
    parser.add_argument('--n_rows', type=int, default=5)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--size', type=int, default=128)
    args = parser.parse_args()

    assert args.output_path.parent.exists()

    show_samples_from_dataset(
        dataset_type=args.dataset_type,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        size=args.size,
        data_path=args.data_path,
        output_path=args.output_path
    )