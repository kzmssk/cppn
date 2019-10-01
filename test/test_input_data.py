from cppn.input_data import InputData
import numpy

def test_input_data():
    data = InputData(width=12, height=14, z_size=2)
    x, z = data.as_batch()
    
    # check shape
    assert x.shape == (12 * 14, 3)
    assert z.shape == (12 * 14, 2)