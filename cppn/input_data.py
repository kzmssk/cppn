import typing
import dataclasses
import numpy
from scipy.stats import truncnorm


def truncated_normal(x, a=-1, b=1, mean=0.5, std=0.2):
    """ Normal dist. [a, b] """
    return truncnorm.pdf(x, (a - mean) / std, (b - mean) / std, loc=mean, scale=std).astype(numpy.float32)


def sample_z(z_size):
    return truncated_normal(numpy.random.rand(z_size))


def interp_z(z1, z2, n_points):
    """ Linear interpolation between z1 and z2 """
    return [a * z1 + (1.0 - a) * z2 for a in numpy.linspace(0, 1, n_points)]


@dataclasses.dataclass
class InputData:
    """ Helper class to generate inputs for CPPN """
    width: int
    height: int
    z_size: int

    def __post_init__(self):
        """ Initialize x, y, and r """
        assert self.width > 0
        assert self.height > 0
        self.x, self.y = numpy.meshgrid(
            numpy.linspace(0, 1, self.width).astype(numpy.float32),
            numpy.linspace(0, 1, self.height).astype(numpy.float32))
        self.r = numpy.sqrt(numpy.square(self.x) + numpy.square(self.y))

    def as_batch(self, z=None):
        """ Return x, y, r, z as batch [width * height, 3 + z_size] """
        if z is None:
            z = sample_z(self.z_size)
        assert z.shape == (self.z_size, )

        return numpy.concatenate((self.x.reshape(-1, 1), self.y.reshape(-1, 1), self.r.reshape(-1, 1)),
                                 axis=1), numpy.tile(z, (self.width * self.height, 1))
