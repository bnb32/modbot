"""Test preprocessing methods"""

import os
import pandas as pd
import numpy as np

from modbot import TEST_DATA_DIR
from modbot.training.data_handling import WeightedGenerator
from modbot.preprocessing import LogCleaning
from modbot.environment import RunConfig, ProcessingConfig


def test_data_generator():
    """Test data generator"""

    data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
    data = pd.read_csv(data_file)

    batch_size = 16
    offensive_weight = 0.2
    kwargs = {'batch_size': batch_size, 'offensive_weight': offensive_weight}
    idx = len(data['text']) % batch_size
    texts = data['text'][:-idx]
    labels = data['is_offensive'][:-idx]
    assert len(texts) % batch_size == 0
    assert len(labels) % batch_size == 0
    gen = WeightedGenerator(texts, labels, **kwargs)

    for batch in gen:
        assert batch[0].shape[0] == batch_size
        assert batch[1].shape[0] == batch_size

    zeros = list(gen.Y).count(0)
    ones = list(gen.Y).count(1)
    frac = ones / (zeros + ones)
    assert np.allclose(frac, offensive_weight, atol=0.01)


def test_log_cleaning():
    """Test log cleaning and memory building"""

    run_config = RunConfig()
    run_config.NICKNAME = 'drchessgremlin'
    proc_config = ProcessingConfig(run_config=run_config)
    log_file = os.path.join(TEST_DATA_DIR, 'test_chat.log')
    log_cleaner = LogCleaning(proc_config)
    ban_lines = []
    for line in log_cleaner.read_log(log_file):
        line_start = f'MOD_ACTION: {run_config.NICKNAME} (delete'
        if line_start in line:
            line = line.replace(line_start, '')
            line = line.strip('\n')
            ban_lines.append(line)
    log_cleaner.clean_log(log_file)
    memory = log_cleaner.mem.chunks
    for user in memory:
        for m in memory[user]:
            if m['mod'] == run_config.NICKNAME and m['deleted']:
                for _, line in enumerate(ban_lines):
                    if m['raw_msg'] in line:
                        ban_lines.remove(line)
                        break
    assert len(ban_lines) == 0
