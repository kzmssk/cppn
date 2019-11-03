import chainer
import numpy
from PIL import Image
import gzip
from pathlib import Path
from cppn.input_data import InputData


class EMnistDataset(chainer.dataset.DatasetMixin):
    def __init__(self, width: int, height: int, z_size: int, data_path: Path):
        self.width = width
        self.height = height
        self.z_size = z_size
        self.input_data = InputData(self.width, self.height, self.z_size)

        # load image data
        with gzip.open(data_path, 'rb') as f:
            self.data = numpy.frombuffer(f.read(), numpy.uint8, offset=16).reshape(-1, 28 * 28)

        super(EMnistDataset, self).__init__()

    def __len__(self):
        return len(self.data)

    def get_example(self, i: int):
        """ Return batch of image [1, 1, S, S], where S = `size` """
        x, z = self.input_data.as_batch()
        
        c = self.data[i]  # [0, 255]
        c = c * -1 + 255  # flip
        c = c.reshape((28, 28)).T
        image = Image.fromarray(c).resize((self.width, self.height))        
        c = numpy.asarray(image).astype(numpy.float32) / 255.0  # 2D array
        c = c.reshape((1, 1, self.width, self.height))
        return x, z, c