"""Module for model training"""
import os
import glob
import pprint

from modbot.utilities.logging import get_logger
from modbot.training.training import (get_last_log, clean_log,
                                            vectorize_and_train)
from modbot.training import training_argparse
from modbot.training.training import recreate_log
from modbot.environment import RunConfig


def append_file(infile, outfile):
    """Append outfile with all but first line from infile"""
    with open(outfile, 'a+') as f1:
        with open(infile, 'r') as f2:
            for i, line in enumerate(f2.read()):
                if i > 0:
                    f1.write(line)


if __name__ == '__main__':
    parser = training_argparse()
    args = parser.parse_args()
    config = RunConfig(args=args)

    logger = get_logger(config.LOGGER_LEVEL)
    logger.info('Running with configuration:\n '
                f'{pprint.pformat(config.public_attrs, indent=1)}')
    logger.info('Saving configuration')
    config.save(config.MODEL_PATH)

    TMP_DIR = os.path.join(config.DATA_DIR, 'tmp')
    os.makedirs(TMP_DIR, exist_ok=True)
    TMP = os.path.join(TMP_DIR, 'clean_tmp.txt')
    TMP_IN = config.infile

    if (config.append or config.update or config.review_decisions
            or config.rerun):
        TMP_OUT = os.path.join(config.DATA_DIR, f'{config.CHANNEL}_data.csv')

    if config.review_decisions:
        TMP_OUT = os.path.join(config.DATA_DIR,
                               f'{config.CHANNEL}_decisions.csv')

    if (not config.append and not config.update and not config.clean):
        TMP_OUT = TMP_IN

    if config.update:
        TMP_IN = get_last_log(config)
        config.append = config.train = True

    if config.clean or config.append:
        clean_log(config, TMP_IN, TMP)
        logger.info(f'Created {TMP}')
        if config.append:
            append_file(TMP, TMP_OUT)
            logger.info(f'Appended {TMP} to {TMP_OUT}')
            TMP_IN = TMP_OUT

    elif config.rerun or config.review_decisions:
        LOG_FILES = glob.glob(f'{config.CHATTY_DIR}/*#{config.CHANNEL}.log')
        RAW_LOG = os.path.join(config.CHATTY_DIR, f'{config.CHANNEL}.log')
        recreate_log(config, RAW_LOG, TMP, TMP_OUT, LOG_FILES)
        TMP_IN = TMP_OUT

    if config.train or config.continue_training or config.just_evaluate:
        clean_log(config, TMP_IN, TMP_IN)
        vectorize_and_train(config, data_file=TMP_IN)
