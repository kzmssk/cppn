import tempfile
from pathlib import Path

from cppn.model import ModelConfig
from experiments import train


def test_train():
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_config = train.TrainConfig(train_image_dir_path=(Path(__file__).parent / 'dammy_image_data').resolve(),
                                         max_iter=2,
                                         batch_size=5,
                                         snapshot_iter_interval=2,
                                         display_iter_interval=1,
                                         n_discriminator_update=3,
                                         evaluation_iter_interval=1)
        model_config = ModelConfig(width=64, height=64, n_units_xyrz=10, n_hidden_units=[10, 10], z_size=2)
        train.train(Path(tmp_dir), train_config, model_config)
