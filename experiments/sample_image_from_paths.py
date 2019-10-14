from pathlib import Path
import argparse
import numpy
from PIL import Image

def sample_image_from_paths(target_path: Path, out_path: Path, n_cols: int, n_rows: int, size: int = 64):
    """ Plot n_rows * n_cols images from target_path """
    imgs = []
    with open(target_path) as f:
        line = f.readline()
        while line:
            target = line.split('\n')[0]
            target = Path(target)

            img = Image.open(target)
            img = img.resize((size, size))
            img = img.convert('L')
            img = numpy.asarray(img)

            imgs.append(img)

            if len(imgs) >= n_cols * n_rows: break
            line = f.readline()  # next line
    
    # imgs is list of (size, size) uint8 array
    rows = []
    for i in range(n_rows):
        row = numpy.concatenate([ imgs[n_cols * i + j] for j in range(n_cols)], axis=1)
        rows.append(row)

    imgs = numpy.concatenate(rows, axis=0)
    Image.fromarray(imgs).save(out_path)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_path', type=Path)
    parser.add_argument('out_path', type=Path)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--n_rows', type=int, default=5)
    args = parser.parse_args()


    # check arguments
    assert args.target_path.exists()
    assert args.out_path.parent.exists()

    sample_image_from_paths(args.target_path, args.out_path, args.n_cols, args.n_rows)
