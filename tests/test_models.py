"""Tests for models"""
import tempfile
import os

from modbot.training.models import SVM
from modbot import TEST_DATA_DIR
from modbot.environment import RunConfig, get_model_path


MODELS = [(SVM)]


def test_model():
    """Test model pipeline"""
    data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
    config = RunConfig()
    setattr(config, 'epochs', 1)
    setattr(config, 'offensive_weight', 0.5)
    for MODEL in MODELS:
        model = MODEL.run(data_file, config)
        _ = model.detailed_score()
        prob = model.predict_proba(['fuck you'])[0][1]
        assert 0 <= float(prob) <= 1


def test_model_save_load():
    """Test model loading"""
    with tempfile.TemporaryDirectory() as td:
        data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
        config = RunConfig()
        setattr(config, 'epochs', 1)
        setattr(config, 'offensive_weight', 0.5)
        for MODEL in MODELS:
            model_path = get_model_path(MODEL.__name__, basedir=td)
            data = MODEL.load_data(data_file)
            X, Y = data['text'], data['is_offensive']
            model = MODEL.run(data_file, config)
            model.save(model_path)
            score = model.score(X, Y)
            model = MODEL.load(model_path)
            assert score == model.score(X, Y)
