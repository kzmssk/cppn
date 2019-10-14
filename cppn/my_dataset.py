import chainer
import numpy
from PIL import Image

from cppn.input_data import InputData


class MyDataset(chainer.dataset.DatasetMixin):
    def __init__(self, paths: list, width: int, height: int, z_size: int):
        self.paths = paths
        self.width = width
        self.height = height
        self.z_size = z_size
        self.input_data = InputData(self.width, self.height, self.z_size)
        super(MyDataset, self).__init__()

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):
        """ returns x, z, c """
        # get
        x, z = self.input_data.as_batch()

        # open image and convert to [1, 1, W, H]
        c = Image.open(self.paths[i])
        c = c.resize((self.width, self.height))

        c = c.convert('L')
        c = numpy.asarray(c, dtype=numpy.float32) / 255.  # [0, 255] -> [0, 1]
        c = c.reshape((1, 1, self.width, self.height))  # [1, 1, W, H]

        return x, z, c
