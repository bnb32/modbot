"""Module for model training"""
import os
import pprint
import shutil
import sys

from modbot.utilities.logging import get_logger
from modbot.training.training import (clean_log,
                                      vectorize_and_train)
from modbot.training import training_argparse
from modbot.environment import RunConfig


logger = get_logger()


def append_file(infile, outfile):
    """Append outfile with all but first line from infile"""
    with open(outfile, 'a+') as f1:
        with open(infile, 'r') as f2:
            for i, line in enumerate(f2.read()):
                if i > 0:
                    f1.write(line)


def get_tmp_out(config, TMP):
    """Get out dir based on args"""
    if config.append:
        TMP_OUT = os.path.join(config.DATA_DIR, f'{config.CHANNEL}_data.csv')

    elif config.review_decisions:
        TMP_OUT = os.path.join(config.DATA_DIR,
                               f'{config.CHANNEL}_decisions.csv')

    elif (not config.append and not config.clean):
        TMP_OUT = TMP

    else:
        TMP_OUT = None

    return TMP_OUT


def main():
    """Main training program"""
    parser = training_argparse()
    args = parser.parse_args()
    config = RunConfig(args=args)
    logger.update_level(config.LOGGER_LEVEL)

    msg = ('**Not running anything. Select either -train or -clean or '
           '-continue_training or -just_evaluate**')
    if not (config.clean or config.train or config.continue_training):
        logger.warning(msg)
        sys.exit()

    logger.info('Running with configuration:\n'
                f'{pprint.pformat(config.public_attrs, indent=1)}')
    if config.train:
        logger.info('Saving configuration')
        config.save(config.MODEL_PATH)

    TMP_DIR = os.path.join(config.DATA_DIR, 'tmp')
    os.makedirs(TMP_DIR, exist_ok=True)
    TMP = os.path.join(TMP_DIR, 'clean_tmp.txt')
    if not os.path.samefile(config.infile, TMP):
        logger.info(f'Copying {config.infile} to {TMP}')
        shutil.copy(config.infile, TMP)

    TMP_OUT = get_tmp_out(config, TMP)

    try:
        clean_log(config, TMP, TMP)
    except KeyboardInterrupt:
        logger.info('Exiting log cleaning')

    if config.append:
        append_file(TMP, TMP_OUT)
        logger.info(f'Appended {TMP} to {TMP_OUT}')
        TMP_IN = TMP_OUT
    else:
        TMP_IN = TMP

    if config.train or config.continue_training or config.just_evaluate:
        try:
            vectorize_and_train(config, data_file=TMP_IN)
        except KeyboardInterrupt:
            logger.info('Exiting training program')


if __name__ == '__main__':
    main()
