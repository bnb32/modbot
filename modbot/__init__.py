"""Modbot module"""
import argparse
import os

from modbot.utilities.utilities import none_or_str

BOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_DIR = os.path.join(os.path.dirname(BOT_DIR), 'tests', 'data')
DATA_DIR = os.path.join(os.path.dirname(BOT_DIR), 'data')
LOG_DIR = os.path.join(DATA_DIR, 'logs')
BERT_PREPROCESS = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
BERT_ENCODER = 'https://tfhub.dev/tensorflow/small_bert/'
BERT_ENCODER += 'bert_en_uncased_L-2_H-128_A-2/2'


def modbot_argparse():
    """Parse args for modbot run"""
    parser = argparse.ArgumentParser(description="Run moderation bot")
    parser.add_argument('-config', '-c', type=str, default=None,
                        help='Configuration file')
    parser.add_argument('-model_path', default=None,
                        type=none_or_str, help='Path to model')
    parser.add_argument('-model_type', default=None,
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
