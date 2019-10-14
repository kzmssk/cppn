from pathlib import Path

from cppn.my_dataset import MyDataset


def test_get_example():
    width = 32
    height = 32
    z_size = 2
    paths = list(Path('./test/dammy_image_data').glob('*.jpg'))
    dataset = MyDataset(paths, width=width, height=height, z_size=z_size)

    x, z, c = dataset.get_example(0)

    assert x.shape == (width * height, 3)
    assert z.shape == (width * height, z_size)
    assert c.shape == (1, 1, width, height)
