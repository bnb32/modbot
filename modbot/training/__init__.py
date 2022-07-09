"""Modbot training module"""
import argparse

from modbot.utilities.utilities import none_or_str, none_or_int, none_or_float
from modbot import (DATA_DIR, LOG_DIR, VALID_MODELS, BERT_ENCODER,
                    BERT_PREPROCESS)


def training_argparse():
    """Parse arguments for training runs"""
    parser = argparse.ArgumentParser(description="Update model with new data")
    parser.add_argument('-infile', type=str,
                        help='Input file for training and/or classification.')
    parser.add_argument('-clean', default=False, action='store_true',
                        help='Process infile for future training.')
    parser.add_argument('-train', default=False, action='store_true',
                        help='Vectorize text and train model.')
    parser.add_argument('-append', default=False, action='store_true',
                        help='Append from input file to existing '
                             'classification dataset.')
    parser.add_argument('-running_check', default=False,
                        action='store_true',
                        help='Use model to check messages that meet the lower '
                             'probability threshold (CHECK_PMIN) defined in '
                             'environment variables, which may have been '
                             'missed.')
    parser.add_argument('-review_decisions', default=False,
                        action='store_true',
                        help='Reclassify all decisions made by the bot.')
    parser.add_argument('-model_type', default='SVM',
                        choices=VALID_MODELS,
                        help='Model type to train', type=str)
    parser.add_argument('-model_path', default=None,
                        help='Path to model', type=none_or_str)
    parser.add_argument('-offensive_weight', default=None, type=none_or_float,
                        help='Desired ratio of ones to number of total '
                             'samples.')
    parser.add_argument('-n_batches', default=None, type=none_or_int,
                        help='Number of training batches per epoch.')
    parser.add_argument('-epochs', default=5, type=int,
                        help='Number of training epochs.')
    parser.add_argument('-batch_size', default=32, type=int,
                        help='Number of samples per batch.')
    parser.add_argument('-sample_size', default=None, type=none_or_int,
                        help='Number of total samples to use for training.')
    parser.add_argument('-val_split', default=0.01, type=float,
                        help='Fraction of full dataset used as validation')
    parser.add_argument('-eval_steps', default=100, type=int,
                        help='Number of steps between model evaluations')
    parser.add_argument('-continue_training', default=False,
                        action='store_true',
                        help='Whether to continue training from saved model.')
    parser.add_argument('-just_evaluate', default=False,
                        action='store_true',
                        help='Whether to just evaluate and skip training.')
    parser.add_argument('-log_dir', default=LOG_DIR,
                        type=str, help='Directory to save logs')
    parser.add_argument('-data_dir', default=DATA_DIR,
                        type=str, help='Parent directory for logs')
    parser.add_argument('-bert_preprocess', default=BERT_PREPROCESS,
                        type=str, help='Path to bert preprocess model')
    parser.add_argument('-bert_encoder', default=BERT_ENCODER,
                        type=str, help='Path to bert encoder')
    parser.add_argument('-chatty_dir', default=None,
                        type=none_or_str,
                        help='Path to chatty logs. Used only if updating, '
                             'appending, or rerunning log using '
                             'source=chatty.')
    parser.add_argument('-channel', default=None,
                        type=none_or_str, help='Channel to moderate')
    parser.add_argument('-nickname', default=None,
                        type=none_or_str, help='Name of modbot')
    parser.add_argument('-config', type=str, default=None,
                        help='Configuration file')
    return parser
