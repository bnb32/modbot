"""Training module"""
from datetime import datetime, timedelta
import numpy as np
import os

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

import modbot.preprocessing as pp
from modbot.training.models import get_model_class
from modbot.moderation import Moderation
from modbot.utilities.logging import get_logger

stop_words = ENGLISH_STOP_WORDS
logger = get_logger()

params = dict({'stop_words': None,
               'tokenizer': None, 'cv': 5, 'method': 'sigmoid',
               'min_df': 1, 'max_df': 1.0, 'analyzer': 'char_wb',
               'ngram_range': (1, 8), 'smooth_idf': 1, 'sublinear_tf': 1,
               'max_iter': 10000, 'C': 1})


def recreate_log(config, raw_log, tmp_log, final_log, log_files):
    """Create log file from scratch

    Parameters
    ----------
    config : RunConfig
        Class storing run time configuration parameters
    raw_log : str
        File name for raw log prior to cleaning
    tmp_log : str
        File name for temporary log file
    final_log : str
        File name for final cleaned log
    log_files : list
        List of raw log files
    config : RunConfig
        Class storing run time configuration parameters
    """
    if config.source == 'chatty':
        logger.info(f'Building raw log: {raw_log}')
        os.system(f'rm -f {raw_log}')
        for log_file in log_files:
            os.system(f'cat {log_file} >> {raw_log}')
        config.clean = True
        clean_log(config, raw_log, tmp_log)
        os.system(f'cp {tmp_log} {final_log}')
        logger.info(f'Created {final_log}')
    else:
        config.clean = True
        clean_log(config, config.infile, tmp_log)
        os.system(f'cp {tmp_log} {final_log}')
        logger.info(f'Created {final_log}')


def get_last_log(config):
    """Get the most recent log

    Parameters
    ----------
    config : RunConfig
        Class storing configuration parameters
    """
    date = datetime.today() - timedelta(days=1)
    file_format = '{dir}/{date}_' + f'#{config.CHANNEL}.log'
    filename = ''

    from_chatty = config.source == 'chatty'
    from_logs = config.source == 'logs'
    from_date = config.from_date

    if from_chatty:
        if from_date is not None:
            filename = file_format.format(dir=config.CHATTY_DIR,
                                          date=from_date)
            return filename
        else:
            filename = file_format.format(dir=config.CHATTY_DIR,
                                          date=date.strftime('%Y-%m-%d'))
    elif from_logs:
        if from_date is not None:
            filename = file_format.format(dir=config.LOG_DIR,
                                          date=from_date)
            return filename
        else:
            filename = file_format.format(dir=config.LOG_DIR,
                                          date=date.strftime('%Y-%m-%d'))

    while not os.path.exists(filename):
        date -= timedelta(days=1)
        if from_chatty:
            filename = file_format.format(dir=config.CHATTY_DIR,
                                          date=date.strftime('%Y-%m-%d'))
        elif from_logs:
            filename = file_format.format(dir=config.LOG_DIR,
                                          date=date.strftime('%Y-%m-%d'))
    return filename


def vectorize_and_train(config, data_file):
    """Vectorize data and train model on vectorized data

    Parameters
    ----------
    config : Config
        Class storing configuration parameters
    data_file : str
        Path to data file including texts and targets to use for training
    """

    offensive_weight = 1 / (1 + config.MULTIPLIER)
    MODEL_CLASS = get_model_class(config.MODEL_TYPE)
    if config.continue_training or config.just_evaluate:
        model = MODEL_CLASS.continue_training(
            config.MODEL_PATH, data_file,
            offensive_weight=offensive_weight,
            epochs=config.epochs,
            just_evaluate=config.just_evaluate,
            batch_size=config.batch_size,
            n_batches=config.n_batches,
            sample_size=config.sample_size,
            val_split=config.val_split)
    else:
        model = MODEL_CLASS.run(
            data_file, offensive_weight=offensive_weight,
            epochs=config.epochs,
            batch_size=config.batch_size,
            n_batches=config.n_batches,
            sample_size=config.sample_size,
            model_path=config.MODEL_PATH,
            val_split=config.val_split,
            bert_preprocess=config.bert_preprocess,
            bert_encoder=config.bert_encoder)

    if not config.just_evaluate:
        model.save(config.MODEL_PATH)
        model_dir = (os.path.dirname(config.MODEL_PATH)
                     if not os.path.isdir(config.MODEL_PATH)
                     else config.MODEL_PATH)
        model.detailed_score(out_dir=model_dir)
    else:
        model.detailed_score()

    logger.info('Done')


def grid_search(data, y, model):
    """Do parameter sweep to opitimize hyperparameters

    Parameters
    ----------
    data : list
        List of texts to use for training
    y : list
        List of targets for the corresponding texts
    model : Initialized scikit-learn model

    dict
        Dictionary containing result of grid search
    """
    logger.info('Parameter sweep')
    plist = np.arange(0.4, 1.0, 0.01).tolist()
    param_grid = {'C': plist}
    clf_grid = GridSearchCV(model, param_grid, verbose=1, cv=5, n_jobs=-1)
    clf_grid.fit(data, y)
    logger.info("Best Parameters:\n%s", clf_grid.best_params_)
    logger.info("Best Estimators:\n%s", clf_grid.best_estimator_)
    return clf_grid


def clean_log(config, infile, outfile, filt=False,
              links=False, correct=False, check_phrases=False,
              check_probs=False, bounds=None):
    """Clean raw log file to prepare for training

    Parameters
    ----------
    infile : str
        Path to log file
    outfile : str
        Path to cleaned output file
    config : RunConfig
        Class storing configuration parameters
    filt : bool, optional
        Whether to filter input data for emotes and repetitions,
        by default False
        classifications, by default False
    links : bool, optional
        Whether to label messages with links, by default False
    correct : bool, optional
        Whether to santitize messages. e.g. remove repetitions, usernames, and
        some special characters
    check_phrases : bool, optional
        Whether to check messages for reclassification, by default False
    check_probs : bool, optional
        Whether to check messages for reclassification based on model
        predictions, by default False
    bounds : list, optional
        List specifying start and end index of messages to review,
        by default None
    """

    bounds = [0, 0] if bounds is None else bounds
    wc = (Moderation(config) if config.MODEL_PATH is not None
          and os.path.exists(config.MODEL_PATH)
          and config.running_check else None)

    if config.clean:
        # clean log
        cl = pp.LogCleaning(config, infile, outfile, wc)
        logger.info('Cleaning log')
        cl.clean_log()

    if filt:
        # filter log
        logger.info('Filtering log')
        pp.filter_log(infile, outfile, config)

    if links:
        # label links
        logger.info('Labeling links')
        pp.separate_links(infile, outfile)

    if correct:
        # correct repetitions and some special chars
        logger.info('Correcting log')
        pp.correct_messages(infile, outfile)

    if check_phrases:
        # reclassify phrases
        logger.info('Checking log')
        pp.check_phrases(infile, outfile)

    if check_probs:
        # reclassify phrases
        logger.info('Using model to check log')
        pp.check_probs(config, infile, outfile, bounds, wc)
