"""Tests for models"""
import tempfile
import os

from modbot.training.models import LSTM, SVM
from modbot import TEST_DATA_DIR
from modbot.utilities.logging import get_logger

logger = get_logger()


def test_lstm():
    """Test lstm pipeline"""

    with tempfile.TemporaryDirectory() as td:

        data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
        model_path = os.path.join(td, 'model')
        model = LSTM.run(data_file, model_path=model_path, epochs=2,
                         offensive_weight=0.5)
        model.detailed_score()

        prob = model.predict_proba(['fuck you'])[0][1]
        logger.info(f'Predicted prob: {prob}')
        assert 0 <= float(prob) <= 1


def test_svm():
    """Test lstm pipeline"""

    data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
    model = SVM.run(data_file)
    model.detailed_score()

    prob = model.predict_proba(['fuck you'])[0][1]
    logger.info(f'Predicted prob: {prob}')
    assert 0 <= float(prob) <= 1


def test_lstm_save():
    """Test lstm saving"""
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, 'model')
        data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
        model = LSTM.run(data_file)
        model.save(model_path)

        assert os.path.exists(model_path)
