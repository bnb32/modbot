"""Test preprocessing methods"""

import os
import pandas as pd
import numpy as np

from modbot import TEST_DATA_DIR
from modbot.training.data_handling import DataGenerator


def test_data_generator():
    """Test data generator"""

    data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
    data = pd.read_csv(data_file)

    print(data.shape)

    batch_size = 16
    offensive_weight = 0.2
    kwargs = {'batch_size': batch_size, 'offensive_weight': offensive_weight}
    idx = len(data['text']) % batch_size
    texts = data['text'][:-idx]
    labels = data['is_offensive'][:-idx]
    assert len(texts) % batch_size == 0
    assert len(labels) % batch_size == 0
    gen = DataGenerator(texts, labels, **kwargs)

    for batch in gen:
        assert batch[0].shape[0] == batch_size
        assert batch[1].shape[0] == batch_size

    zeros = list(gen.Y).count(0)
    ones = list(gen.Y).count(1)
    frac = ones / (zeros + ones)
    assert np.allclose(frac, offensive_weight, atol=0.01)
