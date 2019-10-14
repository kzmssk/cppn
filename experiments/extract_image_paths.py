from pathlib import Path
import argparse
import numpy

def extract_image_paths(target_dir_path: Path, out_path: Path, N: int):
    """ Extract N file paths from taret_dir_path """
    n = 0
    with open(out_path, 'w') as f:
        for target in target_dir_path.glob('*.jpg'):
            f.write(str(target) + '\n')
            n += 1
            if n >= N:
                break
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir_path', type=Path)
    parser.add_argument('out_path', type=Path)
    parser.add_argument('--n', type=int, default=10000)
    args = parser.parse_args()

    # check arguments
    assert args.target_dir_path.exists()
    assert args.out_path.parent.exists()

    extract_image_paths(args.target_dir_path, args.out_path, args.n)
