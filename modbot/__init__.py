"""Modbot module"""
import argparse
import os

from modbot.utilities.utilities import none_or_str

BOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_DIR = os.path.join(os.path.dirname(BOT_DIR), 'tests', 'data')
DATA_DIR = os.path.join(os.path.dirname(BOT_DIR), 'data')
LOG_DIR = os.path.join(DATA_DIR, 'logs')
VALID_MODELS = ('BERT', 'BERT_CNN', 'BERT_CNN_LSTM', 'BERT_LSTM_CNN',
                'BERT_LSTM', 'CNN', 'SVM', 'LSTM', 'BERT_CNN_TORCH')
BERT_PREPROCESS = os.path.join(DATA_DIR, 'bert_preprocess')
BERT_ENCODER = os.path.join(DATA_DIR, 'bert_encoder')


def modbot_argparse():
    """Parse args for modbot run"""
    parser = argparse.ArgumentParser(description="Run moderation bot")
    parser.add_argument('-config', type=str, default=None,
                        help='Configuration file')
    parser.add_argument('-model_path', default=None,
                        type=none_or_str, help='Path to model')
    parser.add_argument('-model_type', default='SVM',
                        choices=VALID_MODELS,
                        help='Model type to use for modding', type=str)
    parser.add_argument('-channel', default=None,
                        type=none_or_str, help='Channel to moderate')
    parser.add_argument('-nickname', default=None,
                        type=none_or_str, help='Name of modbot')
    parser.add_argument('-log_dir', default=LOG_DIR,
                        type=str, help='Directory to save logs')
    parser.add_argument('-data_dir', default=DATA_DIR,
                        type=str, help='Parent directory for logs')
    return parser
