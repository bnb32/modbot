"""Tests for models"""
import tempfile
import os
import pytest
import pandas as pd

from modbot.training.models import (LSTM, SVM, CNN, BERT, BertCNN,
                                    BertCnnLstm, BertCnnTorch, BertLSTM)
from modbot import TEST_DATA_DIR
from modbot.utilities.logging import get_logger

logger = get_logger()

MODELS = [(SVM), (LSTM), (CNN), (BERT), (BertCNN), (BertCnnLstm),
          (BertCnnTorch), (BertLSTM)]


@pytest.mark.parametrize('MODEL', MODELS)
def test_model(MODEL):
    """Test model pipeline"""
    with tempfile.TemporaryDirectory() as td:
        data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
        model_path = os.path.join(td, 'model')
        model = MODEL.run(data_file, model_path=model_path, epochs=1,
                          offensive_weight=0.5)
        model.detailed_score()
        prob = model.predict_proba(['fuck you'])[0][1]
        logger.info(f'Predicted prob: {prob}')
        assert 0 <= float(prob) <= 1


@pytest.mark.parametrize('MODEL', MODELS)
def test_model_save_load(MODEL):
    """Test model loading"""
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, 'model')
        data_file = os.path.join(TEST_DATA_DIR, 'test_data.csv')
        data = pd.read_csv(data_file)
        X, Y = data['text'], data['is_offensive']
        model = MODEL.run(data_file, model_path=model_path, epochs=1,
                          offensive_weight=0.5)
        model.save(model_path)
        score = model.score(X, Y)
        model = MODEL.load(model_path)
        assert score == model.score(X, Y)
