import chainer
import numpy
from PIL import Image
from cppn.input_data import InputData


class MnistDataset(chainer.dataset.DatasetMixin):
    def __init__(self, width: int, height: int, z_size: int):
        self.width = width
        self.height = height
        self.z_size = z_size
        self.input_data = InputData(self.width, self.height, self.z_size)

        # use only train dataset
        self.train_data, _ = chainer.datasets.get_mnist()
        super(MnistDataset, self).__init__()
    
    def __len__(self):
        return len(self.train_data)

    def get_example(self, i: int):
        """ Return batch of image [1, 1, S, S], where S = `size` """
        x, z = self.input_data.as_batch()

        c, _ = self.train_data[i]
        c = c * -1 + 1.0  # flip [0, 1] -> [1, 0]
        c = c.reshape((28, 28)) * 255.0
        image = Image.fromarray(c.astype(numpy.uint8)).resize((self.width, self.height))
        c = numpy.asarray(image).astype(numpy.float32) / 255.0  # 2D array
        c = c.reshape((1, 1, self.width, self.height))
        return x, z, c