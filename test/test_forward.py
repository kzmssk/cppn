from cppn.model import CPPN, ModelConfig
from cppn.input_data import InputData


def test_forward():
    width = 5
    height = 7
    z_size = 2

    model = CPPN(ModelConfig(
        width=width,
        height=height,
        n_units_xyrz=3,
        n_hidden_units=[5, 5],
        z_size=z_size
    ))

    x, z = InputData(width=width, height=height, z_size=z_size).as_batch()
    y = model.forward(x, z)
    assert y.shape == (1, width, height)