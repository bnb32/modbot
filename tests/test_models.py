"""Tests for models"""
import tempfile
import os
import pytest
import pandas as pd

from modbot.training.models import SVM
from modbot import TEST_DATA_DIR
from modbot.utilities.logging import get_logger
from modbot.environment import RunConfig, get_model_path

logger = get_logger()

MODELS = [(SVM)]


@pytest.mark.parametrize('MODEL', MODELS)
def test_model(MODEL):
    """Test model pipeline"""
    data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
    config = RunConfig()
    setattr(config, 'epochs', 1)
    setattr(config, 'offensive_weight', 0.5)
    model = MODEL.run(data_file, config)
    _ = model.detailed_score()
    prob = model.predict_proba(['fuck you'])[0][1]
    logger.info(f'Predicted prob: {prob}')
    assert 0 <= float(prob) <= 1


@pytest.mark.parametrize('MODEL', MODELS)
def test_model_save_load(MODEL):
    """Test model loading"""
    with tempfile.TemporaryDirectory() as td:
        data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
        model_path = get_model_path(MODEL.__name__, basedir=td)
        data = pd.read_csv(data_file)
        X, Y = data['text'], data['is_offensive']
        config = RunConfig()
        setattr(config, 'epochs', 1)
        setattr(config, 'offensive_weight', 0.5)
        model = MODEL.run(data_file, config)
        model.save(model_path)
        score = model.score(X, Y)
        model = MODEL.load(model_path)
        assert score == model.score(X, Y)
