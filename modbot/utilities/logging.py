"""Logging module for storing chat and connection info"""
import os
from emoji import demojize
import logging
from sys import stdout

from modbot.utilities.utilities import is_user_type_irc


VERBOSE_LEVEL = logging.INFO - 2
EXTRA_VERBOSE_LEVEL = logging.INFO - 4
CHAT_LEVEL = logging.INFO + 2
MOD_LEVEL = logging.INFO + 4
PRIVATE_LEVEL = logging.INFO - 10


def extra_verbose(self, message, *args, **kws):
    """Extra verbose log level"""
    self._log(EXTRA_VERBOSE_LEVEL, message, args, **kws)


def verbose(self, message, *args, **kws):
    """Verbose log level"""
    self._log(VERBOSE_LEVEL, message, args, **kws)


def chat(self, message, *args, **kws):
    """Chat log level"""
    self._log(CHAT_LEVEL, message, args, **kws)


def mod(self, message, *args, **kws):
    """MOD log level"""
    self._log(MOD_LEVEL, message, args, **kws)


def private(self, message, *args, **kws):
    """MOD log level"""
    self._log(PRIVATE_LEVEL, message, args, **kws)


def get_logger(level=18):
    """Initialize logger

    Returns
    -------
    logger
        Logger object
    """
    logger = logging.getLogger('modbot_logger')

    if not logger.handlers:
        logging.addLevelName(EXTRA_VERBOSE_LEVEL, 'EXTRA_VERBOSE')
        logging.addLevelName(VERBOSE_LEVEL, 'VERBOSE')
        logging.addLevelName(CHAT_LEVEL, 'CHAT')
        logging.addLevelName(MOD_LEVEL, 'MOD')

        logging.Logger.extra_verbose = extra_verbose
        logging.Logger.verbose = verbose
        logging.Logger.chat = chat
        logging.Logger.mod = mod

        logger.setLevel(level)
        sh = logging.StreamHandler(stdout)
        sh.setLevel(level)
        formatter = logging.Formatter(
            '[%(levelname)s] %(filename)s:%(lineno)d %(asctime)s %(message)s')
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.propagate = False
    return logger


class Logging:
    """Log handling class"""
    def __init__(self, run_config):
        self.log_path = run_config.LOG_PATH
        self.write_log = run_config.WRITE_LOG
        self.logger = None
        if self.write_log:
            self.logger = self.initialize_logger(run_config)
        self.UserLog = {}

    def initialize_logger(self, run_config):
        """Initialize private logger

        Parameters
        ----------
        run_config : RunConfig
            RunConfig class storing run time configuration parameters

        Returns
        -------
        logger
            Private logger used to write chat messages and mod actions
        """
        logger = get_logger()
        os.makedirs(run_config.LOG_DIR, exist_ok=True)
        private_logger = logging.getLogger('private_logger')
        if not private_logger.handlers:
            logging.Logger.private = private
            logging.addLevelName(PRIVATE_LEVEL, 'PRIVATE')
            logger.info('Initializing private logger for chat and mod actions')
            private_logger.setLevel(level='PRIVATE')
            logFormatter = logging.Formatter("%(message)s")
            fileHandler = logging.FileHandler(self.log_path)
            fileHandler.setFormatter(logFormatter)
            private_logger.addHandler(fileHandler)
            private_logger.propagate = False
        return private_logger

    @staticmethod
    def get_value(key, dictionary):
        """Search nested dictionary for key

        Parameters
        ----------
        key : str
            Key to search dictionary for
        dictionary : dict
            Dictionary to search key for

        Returns
        -------
        str
            Value in dictionary corresponding to key, or empty string if not
            found
        """
        if key in dictionary:
            return dictionary[key]
        else:
            for k in dictionary:
                if key in k:
                    return dictionary[k]
            return ''

    def append_log(self, line):
        """Add entry to log

        Parameters
        ----------
        line : str
            Line to write to log after some sanitizing
        """
        write_check = (not is_user_type_irc('vip', line)
                       and not is_user_type_irc('mod', line)
                       and self.write_log)
        if write_check:
            self.logger.private(line.encode('ascii', 'ignore').decode())

    @staticmethod
    def build_chat_log_entry(info):
        """Build standard chat message log entry

        Parameters
        ----------
        info : dict
            Dictionary of info from IRC message

        Returns
        -------
        str
            Chat log entry to write to log
        """
        log_entry = "<"
        if info['isMod']:
            log_entry += "@"
        if info['isVip']:
            log_entry += "!"
        if info['isSub']:
            log_entry += "%"
        if info['isPartner']:
            log_entry += "~"
        log_entry += info['user'] + "> "
        log_entry += info['msg']

        return log_entry

    @staticmethod
    def build_action_log_entry(action, user, moderator, msg, secs, msg_id):
        """Build action log entry from message content

        Parameters
        ----------
        action : str
            Moderation action. e.g ban, timeout, delete
        user : str
            Username
        moderator : str
            Moderator username who performed action
        msg : str
            The message which preceeded the mod action
        secs : str
            Timeout length in seconds
        msg_id : str
            Message id

        Returns
        -------
        str
            Log entry containing moderation action info
        """
        log_entry = ""
        if "ban" in action:
            log_entry = f"BAN: {user}\n"
            log_entry += f"MOD_ACTION: {moderator}"
            log_entry += f" ({action} {user} {msg})"
        elif "timeout" in action:
            log_entry = f"BAN: {user} ({secs}s)\n"
            log_entry += f"MOD_ACTION: {moderator}"
            log_entry += f" ({action} {user} {secs} {msg})"
        elif "delete" in action:
            log_entry = f"DELETED: {user} ({msg})\n"
            log_entry += f"MOD_ACTION: {moderator}"
            log_entry += f" ({action} {user}"
            log_entry += f" {msg} {msg_id})"
        else:
            log_entry += f"MOD_ACTION: {moderator}"
            log_entry += f" ({action} {user})"
        return log_entry

    def get_info_from_pubsub(self, msg_dict):
        """Get info from PubSub message for logging

        Parameters
        ----------
        msg_dict : dict
            Dictionary containing PubSub message info

        Returns
        -------
        action : str
            Moderation action. e.g ban, timeout, delete
        user : str
            Username
        moderator : str
            Moderator username who performed action
        msg : str
            The message which preceeded the mod action
        secs : str
            Timeout length in seconds
        msg_id : str
            Message id
        """
        action = self.get_value('moderation_action', msg_dict)
        moderator = self.get_value('created_by', msg_dict)
        user = msg = secs = msg_id = ''
        if 'args' in msg_dict:
            user = msg_dict['args'][0]
        else:
            user = self.get_value('target_user_login', msg_dict)

        if action in ['ban', 'timeout']:
            if user in self.UserLog:
                msg = self.UserLog[user]
            else:
                msg = ''
        if "timeout" in action:
            secs = msg_dict['args'][1]
        if "delete" in action:
            msg = demojize(msg_dict['args'][1])
            msg_id = msg_dict['args'][2]
        return action, user, moderator, msg, secs, msg_id